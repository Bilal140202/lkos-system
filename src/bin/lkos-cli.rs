//! lkos-cli — command-line interface to the LKOS engine.
//!
//! Zero-dependency argument parsing (documented choice: the CLI is a thin
//! wrapper; the engine API is the product).

use lkos::{Config, Lkos, QueryRequest, RetrievalMode};
use std::io::Write;

fn usage() -> &'static str {
    "lkos-cli — Local Knowledge Object System

USAGE:
    lkos-cli [--db <path>] <COMMAND> [ARGS]

COMMANDS:
    init                        Create an empty knowledge base
    ingest <file>...            Ingest documents (txt/md/code/data)
    docs                        List documents
    search <query> [-k N]       Hybrid retrieval with explanations
    ask <question>              Grounded answer (requires --llm)
    entities [doc_id]           Entity index / entities of a document
    claims [doc_id]             Claims / claims of a document
    conflicts [N]               Detected numeric conflicts
    graph <entity_name>         One-hop entity neighborhood
    stats                       Library statistics
    provenance <doc_id>         Provenance export for a document
    delete <doc_id>             Delete a document and derived knowledge
    backup <dest>               Online backup of the database
    integrity                   Run SQLite integrity check

OPTIONS:
    --db <path>                 Knowledge base path (default ./library.lkos)
    --json                      JSON output where supported
    --llm <binary>              llama.cpp-compatible binary for `ask`
    --model <gguf>              Model path for `ask`
    -h, --help                  This help
"
}

struct Args {
    db: String,
    json: bool,
    llm: Option<String>,
    model: Option<String>,
    rest: Vec<String>,
}

fn parse_args(mut args: std::env::Args) -> Args {
    let mut out = Args {
        db: "library.lkos".into(),
        json: false,
        llm: None,
        model: None,
        rest: Vec::new(),
    };
    let _ = args.next(); // program name
    while let Some(a) = args.next() {
        match a.as_str() {
            "--db" => out.db = args.next().unwrap_or_else(|| out.db.clone()),
            "--json" => out.json = true,
            "--llm" => out.llm = Some(args.next().unwrap_or_default()),
            "--model" => out.model = Some(args.next().unwrap_or_default()),
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            _ => out.rest.push(a),
        }
    }
    out
}

fn main() {
    let args = parse_args(std::env::args());
    let Some(cmd) = args.rest.first().cloned() else {
        eprint!("{}", usage());
        std::process::exit(2);
    };
    let rest: Vec<String> = args.rest.iter().skip(1).cloned().collect();

    let config = Config::default();
    if let Err(e) = run(&cmd, &rest, &args, config) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(
    cmd: &str,
    rest: &[String],
    args: &Args,
    config: Config,
) -> lkos::Result<()> {
    match cmd {
        "init" => {
            let _ = Lkos::open(&args.db, config)?;
            println!("initialized knowledge base at {}", args.db);
            Ok(())
        }
        "ingest" => {
            let engine = Lkos::open(&args.db, config)?;
            let mut n = 0usize;
            for path in rest {
                match engine.ingest_file(path) {
                    Ok(doc) => {
                        println!(
                            "ingested: {} (id {}, {} chunks, state {})",
                            doc.filename, doc.id, doc.chunk_count, doc.readiness_state
                        );
                        n += 1;
                    }
                    Err(e) => eprintln!("skipped {path}: {e}"),
                }
            }
            println!("{n} document(s) ingested");
            Ok(())
        }
        "docs" => {
            let engine = Lkos::open(&args.db, config)?;
            for d in engine.documents()? {
                println!(
                    "{:>4}  {:<24} {:<10} {:>4} chunks  [{}]",
                    d.id, d.filename, d.doc_type, d.chunk_count, d.readiness_state
                );
            }
            Ok(())
        }
        "search" => {
            let engine = Lkos::open(&args.db, config)?;
            let mut query = String::new();
            let mut k = 8usize;
            let mut i = 0usize;
            while i < rest.len() {
                match rest[i].as_str() {
                    "-k" => {
                        k = rest.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(8);
                        i += 2;
                    }
                    _ => {
                        query.push_str(&rest[i]);
                        query.push(' ');
                        i += 1;
                    }
                }
            }
            let req = QueryRequest::new(query).top_k(k).mode(RetrievalMode::Auto);
            let resp = engine.query(req)?;
            println!("plan: {}", resp.plan_explanation);
            println!();
            for h in &resp.hits {
                let matched: Vec<String> = h
                    .matched_by
                    .iter()
                    .map(|m| match m {
                        lkos::MatchSource::Vector { rank, cosine } => {
                            format!("vector#{rank}({cosine:.3})")
                        }
                        lkos::MatchSource::Fts { rank, .. } => format!("bm25#{rank}"),
                        lkos::MatchSource::Entity { name } => format!("entity({name})"),
                        lkos::MatchSource::Phrase => "phrase".into(),
                        lkos::MatchSource::SectionTitle => "section".into(),
                    })
                    .collect();
                println!(
                    "[{}] {:.4} {} :: {} ({})",
                    h.rank,
                    h.score,
                    h.document,
                    h.section.as_deref().unwrap_or("-"),
                    matched.join(", ")
                );
                let snippet: String = h.text.chars().take(220).collect();
                println!("     {snippet}");
                println!();
            }
            Ok(())
        }
        "ask" => {
            let engine = Lkos::open(&args.db, config)?;
            if let (Some(binary), Some(model)) = (&args.llm, &args.model) {
                let provider = lkos::llm::LlamaCppProvider::new(binary, model, engine.config().llm_timeout_secs);
                engine.set_llm(std::sync::Arc::new(provider));
            } else {
                engine.set_llm(std::sync::Arc::new(lkos::llm::NullProvider));
            }
            let question = rest.join(" ");
            let answer = engine.ask(&question)?;
            println!("{answer}");
            Ok(())
        }
        "entities" => {
            let engine = Lkos::open(&args.db, config)?;
            if let Some(doc_id) = rest.first().and_then(|v| v.parse::<i64>().ok()) {
                for e in engine.document_entities(doc_id)? {
                    println!("{:>4}  {:<28} {:<12} {} mentions", e.id, e.display_name, e.entity_type, e.mention_count);
                }
            } else {
                for e in engine.list_entities(100)? {
                    println!("{:>4}  {:<28} {:<12} {} mentions", e.id, e.display_name, e.entity_type, e.mention_count);
                }
            }
            Ok(())
        }
        "claims" => {
            let engine = Lkos::open(&args.db, config)?;
            if let Some(doc_id) = rest.first().and_then(|v| v.parse::<i64>().ok()) {
                for c in engine.document_claims(doc_id)? {
                    println!(
                        "[{}] {} --{}--> {} (conf {:.2}, from {:?})",
                        c.id, c.subject, c.predicate, c.object, c.confidence, c.valid_from
                    );
                }
            }
            Ok(())
        }
        "conflicts" => {
            let engine = Lkos::open(&args.db, config)?;
            let limit = rest.first().and_then(|v| v.parse().ok()).unwrap_or(20);
            for cf in engine.conflicts(limit)? {
                println!(
                    "[{}] {} x {}: {:.0}% delta — {}",
                    cf.id, cf.claim_a, cf.claim_b, cf.delta * 100.0, cf.explanation
                );
            }
            Ok(())
        }
        "graph" => {
            let engine = Lkos::open(&args.db, config)?;
            let name = rest.join(" ");
            let entity_id = engine
                .entity_id_by_name(&name)?
                .ok_or_else(|| lkos::LkosError::Other(format!("entity '{name}' not found")))?;
            let nb = engine.neighborhood(entity_id, 15)?;
            println!("{} [{}] — {} documents", nb.center.name, nb.center.entity_type, nb.documents.len());
            for (n, e) in &nb.neighbors {
                println!("  --{}(w={:.0})--> {} [{}]", e.relationship, e.weight, n.name, n.entity_type);
            }
            Ok(())
        }
        "stats" => {
            let engine = Lkos::open(&args.db, config)?;
            let s = engine.stats()?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            } else {
                println!("{:<22} {}", "documents", s.documents);
                println!("{:<22} {}", "chunks", s.chunks);
                println!("{:<22} {}", "entities", s.entities);
                println!("{:<22} {}", "entity_mentions", s.entity_mentions);
                println!("{:<22} {}", "claims", s.claims);
                println!("{:<22} {}", "conflicts", s.conflicts);
                println!("{:<22} {}", "relationships", s.relationships);
                println!("{:<22} {}", "provenance_records", s.provenance_records);
                println!("{:<22} {} bytes", "db_size", s.db_size_bytes);
            }
            Ok(())
        }
        "provenance" => {
            let doc_id = rest
                .first()
                .and_then(|v| v.parse::<i64>().ok())
                .ok_or_else(|| lkos::LkosError::InvalidInput("provenance <doc_id>".into()))?;
            let store = lkos::storage::Store::open(&args.db)?;
            let prov = lkos::provenance::export_document(&store, doc_id)?;
            match prov {
                Some(p) => println!("{}", serde_json::to_string_pretty(&p).unwrap_or_default()),
                None => println!("document not found"),
            }
            Ok(())
        }
        "delete" => {
            let engine = Lkos::open(&args.db, config)?;
            let id = rest
                .first()
                .and_then(|v| v.parse::<i64>().ok())
                .ok_or_else(|| lkos::LkosError::InvalidInput("delete <doc_id>".into()))?;
            engine.delete_document(id)?;
            println!("deleted document {id}");
            Ok(())
        }
        "backup" => {
            let dest = rest
                .first()
                .cloned()
                .ok_or_else(|| lkos::LkosError::InvalidInput("backup <dest>".into()))?;
            let engine = Lkos::open(&args.db, config)?;
            engine.backup_to(&dest)?;
            println!("backup written to {dest}");
            Ok(())
        }
        "integrity" => {
            let engine = Lkos::open(&args.db, config)?;
            for line in engine.integrity_check()? {
                println!("{line}");
            }
            Ok(())
        }
        other => {
            eprintln!("unknown command: {other}");
            eprint!("{}", usage());
            std::process::exit(2);
        }
    }
}

// Silence unused import when features change.
#[allow(dead_code)]
fn _silence(_: &mut dyn Write) {}
