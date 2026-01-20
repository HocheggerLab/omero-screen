# Architectural Decision Records (ADRs)

Document key architectural and technical decisions with context and trade-offs.

## Format

```markdown
### ADR-XXX: Decision Title (YYYY-MM-DD)

**Context:**
- Why the decision was needed
- What problem it solves

**Decision:**
- What was chosen

**Alternatives Considered:**
- Option 1 -> Why rejected
- Option 2 -> Why rejected

**Consequences:**
- Benefits
- Trade-offs
```

## Examples

### ADR-001: Use Cellpose for Cell Segmentation (2024-12-01)

**Context:**
- Need robust, automated cell and nucleus segmentation for high-content screening
- Must handle various cell lines and imaging conditions
- Team lacks expertise to train segmentation models from scratch

**Decision:**
- Use Cellpose pre-trained models (nucleus and cyto2)
- Select models automatically based on cell line and magnification metadata

**Alternatives Considered:**
- StarDist -> Good for nuclei, but limited cell segmentation
- Custom U-Net -> Requires extensive training data and expertise
- CellProfiler -> Less robust for varied imaging conditions

**Consequences:**
- Benefits: High-quality segmentation out of box, actively maintained, supports custom models
- Trade-offs: Dependency on external library, slower than simpler threshold-based methods

### ADR-002: Use DuckDB for Single-Cell Data Storage (2024-11-15)

**Context:**
- Need fast local querying of millions of single-cell measurements
- CSV files become unwieldy for large screening campaigns
- Don't want overhead of PostgreSQL for local analysis

**Decision:**
- Use DuckDB embedded database for cellview package
- Organize by project → experiment → plate → condition hierarchy

**Alternatives Considered:**
- SQLite -> Slower for analytical queries, less optimized for large datasets
- PostgreSQL -> Overkill for local storage, requires server management
- Parquet files -> Good for storage, but need query layer on top

**Consequences:**
- Benefits: Fast analytical queries, SQL interface, single file database, no server needed
- Trade-offs: Newer technology with smaller ecosystem than SQLite/Postgres

---

## Active Decisions

(Add new ADRs below this line, incrementing the ADR number)
