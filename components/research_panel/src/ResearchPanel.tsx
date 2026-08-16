/* ============================================================
   ResearchPanel.tsx
   Apple Dark Blue Aesthetic — ArXiv Deep-Research Agent
   ============================================================ */

import React, { useEffect, useState, useRef, useCallback } from "react"
import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import "./ResearchPanel.css"

/* ── Types ──────────────────────────────────────────────── */
interface Benchmark {
  task_or_dataset: string
  base_model: string
  speedup_factor: string | null
  accuracy_delta: string | null
}

interface Paper {
  arxiv_id: string
  title: string
  authors: string[]
  abstract: string
  published_date: string
  arxiv_url: string
  primary_category: string
  hf_upvotes: number
  citation_count: number
  github_urls: string[]
}

interface Analysis {
  arxiv_id: string
  title: string
  core_innovation: string
  architecture_details: string
  mathematical_formulation: string
  reproducibility_notes: string
  benchmarks: Benchmark[]
  limitations: string[]
}

interface PanelArgs {
  query: string
  papers: Paper[]
  analyses: Analysis[]
  report: string
  fact_check_passed: boolean
  saved_path: string
  is_loading: boolean
  status_steps: string[]
}

/* ── Icons (inline SVG as React components) ─────────────── */
const IconSearch = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
  </svg>
)

const IconBook = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
  </svg>
)

const IconFlask = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 3h6l3 7H6L9 3z"/><path d="M6 10l-2 8a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1l-2-8"/>
    <path d="M10 15h4"/>
  </svg>
)

const IconCheck = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
)

const IconLoader = () => (
  <svg className="spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
  </svg>
)

const IconExternalLink = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
    <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
  </svg>
)

const IconDownload = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
)

/* ── Sub-components ─────────────────────────────────────── */
const StatusTimeline: React.FC<{ steps: string[] }> = ({ steps }) => (
  <div className="status-timeline">
    {steps.map((step, i) => (
      <div key={i} className="status-step">
        <div className="step-dot">
          <IconCheck />
        </div>
        <span className="step-text">{step}</span>
      </div>
    ))}
    <div className="status-step status-step--loading">
      <div className="step-dot step-dot--loading">
        <IconLoader />
      </div>
      <span className="step-text">Processing…</span>
    </div>
  </div>
)

const MetricPill: React.FC<{ label: string; value: string | number }> = ({ label, value }) => (
  <div className="metric-pill">
    <span className="metric-pill__label">{label}</span>
    <span className="metric-pill__value">{value}</span>
  </div>
)

const PaperCard: React.FC<{ paper: Paper; onSelect: (id: string) => void; selected: boolean }> = ({
  paper, onSelect, selected,
}) => (
  <div
    className={`paper-card ${selected ? "paper-card--selected" : ""}`}
    onClick={() => onSelect(paper.arxiv_id)}
    role="button"
    tabIndex={0}
    onKeyDown={(e) => e.key === "Enter" && onSelect(paper.arxiv_id)}
  >
    <div className="paper-card__header">
      <span className="paper-card__id">arXiv:{paper.arxiv_id}</span>
      <span className="paper-card__category">{paper.primary_category}</span>
    </div>
    <h3 className="paper-card__title">{paper.title}</h3>
    <p className="paper-card__authors">{paper.authors.slice(0, 3).join(", ")}{paper.authors.length > 3 ? " et al." : ""}</p>
    <p className="paper-card__abstract">{paper.abstract.slice(0, 180)}…</p>
    <div className="paper-card__footer">
      <MetricPill label="⭐ HF" value={paper.hf_upvotes} />
      <MetricPill label="📚 Citations" value={paper.citation_count ?? 0} />
      <MetricPill label="📅 Published" value={paper.published_date?.slice(0, 7) ?? "N/A"} />
      <a
        href={paper.arxiv_url}
        target="_blank"
        rel="noopener noreferrer"
        className="arxiv-link"
        onClick={(e) => e.stopPropagation()}
      >
        View <IconExternalLink />
      </a>
    </div>
  </div>
)

const BenchmarkTable: React.FC<{ benchmarks: Benchmark[] }> = ({ benchmarks }) => {
  if (!benchmarks.length) return <p className="empty-state">No benchmarks extracted.</p>
  return (
    <div className="table-wrapper">
      <table className="benchmark-table">
        <thead>
          <tr>
            <th>Dataset / Task</th>
            <th>Base Model</th>
            <th>Speedup</th>
            <th>Accuracy Δ</th>
          </tr>
        </thead>
        <tbody>
          {benchmarks.map((b, i) => (
            <tr key={i}>
              <td><code>{b.task_or_dataset}</code></td>
              <td>{b.base_model}</td>
              <td>{b.speedup_factor ?? "—"}</td>
              <td>{b.accuracy_delta ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const AnalysisCard: React.FC<{ analysis: Analysis }> = ({ analysis }) => {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="analysis-card">
      <div className="analysis-card__header" onClick={() => setExpanded(!expanded)}>
        <span className="analysis-card__id">arXiv:{analysis.arxiv_id}</span>
        <h3 className="analysis-card__title">{analysis.title}</h3>
        <span className="analysis-card__toggle">{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="analysis-card__body">
          <div className="analysis-section">
            <div className="analysis-label">💡 Core Innovation</div>
            <p>{analysis.core_innovation}</p>
          </div>
          <div className="analysis-section">
            <div className="analysis-label">🏗 Architecture</div>
            <p>{analysis.architecture_details}</p>
          </div>
          {analysis.mathematical_formulation && (
            <div className="analysis-section">
              <div className="analysis-label">🧮 Mathematical Formulation</div>
              <pre className="math-block">{analysis.mathematical_formulation}</pre>
            </div>
          )}
          {analysis.limitations?.length > 0 && (
            <div className="analysis-section">
              <div className="analysis-label">⚠️ Limitations</div>
              <ul className="limitations-list">
                {analysis.limitations.map((l, i) => <li key={i}>{l}</li>)}
              </ul>
            </div>
          )}
          <div className="analysis-section">
            <div className="analysis-label">📊 Benchmarks ({analysis.benchmarks?.length ?? 0})</div>
            <BenchmarkTable benchmarks={analysis.benchmarks ?? []} />
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Main Component ─────────────────────────────────────── */
type Tab = "report" | "papers" | "analyses"

const ResearchPanel: React.FC<ComponentProps> = (props) => {
  const args = props.args as PanelArgs
  const {
    query = "",
    papers = [],
    analyses = [],
    report = "",
    fact_check_passed = false,
    saved_path = "",
    is_loading = false,
    status_steps = [],
  } = args

  const [activeTab, setActiveTab] = useState<Tab>("report")
  const [selectedPaper, setSelectedPaper] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-resize iframe to fit content
  const updateHeight = useCallback(() => {
    if (containerRef.current) {
      Streamlit.setFrameHeight(containerRef.current.scrollHeight + 32)
    }
  }, [])

  useEffect(() => {
    Streamlit.setFrameHeight()
  })

  useEffect(() => {
    const ro = new ResizeObserver(updateHeight)
    if (containerRef.current) ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [updateHeight])

  useEffect(() => {
    updateHeight()
  }, [activeTab, selectedPaper, is_loading, updateHeight])

  // Send selected paper back to Python
  const handlePaperSelect = (id: string) => {
    const s = id === selectedPaper ? null : id
    setSelectedPaper(s)
    Streamlit.setComponentValue({ event: "paper_selected", arxiv_id: s })
  }

  // Download report
  const handleDownload = () => {
    const blob = new Blob([report], { type: "text/markdown" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `arxiv_report_${Date.now()}.md`
    a.click()
    URL.revokeObjectURL(url)
    Streamlit.setComponentValue({ event: "report_downloaded", query })
  }

  return (
    <div className="rp-root" ref={containerRef}>
      {/* ── Header ── */}
      <div className="rp-header">
        <div className="rp-header__brand">
          <div className="rp-brand-icon">
            <IconSearch />
          </div>
          <div>
            <div className="rp-header__title">ArXiv Deep-Research Agent</div>
            <div className="rp-header__subtitle">Autonomous Multi-Agent LLM Research Pipeline</div>
          </div>
        </div>
        {query && (
          <div className="rp-header__query">
            <span className="query-badge">
              <IconSearch />
              {query}
            </span>
            {fact_check_passed && (
              <span className="fact-badge fact-badge--pass">
                <IconCheck /> Fact-Checked
              </span>
            )}
          </div>
        )}
      </div>

      {/* ── Loading State ── */}
      {is_loading && (
        <div className="rp-loading-panel">
          <div className="rp-loading-panel__spinner">
            <div className="glow-ring" />
            <IconLoader />
          </div>
          <div className="rp-loading-panel__title">Agent Running…</div>
          <StatusTimeline steps={status_steps} />
        </div>
      )}

      {/* ── Results State ── */}
      {!is_loading && (papers.length > 0 || report) && (
        <>
          {/* Stats bar */}
          <div className="stats-bar">
            <MetricPill label="Papers" value={papers.length} />
            <MetricPill label="Analyses" value={analyses.length} />
            <MetricPill label="Steps" value={status_steps.length} />
            {saved_path && (
              <span className="saved-badge">
                <IconCheck /> Saved to reports/
              </span>
            )}
          </div>

          {/* Tab bar */}
          <div className="tab-bar">
            <button
              className={`tab-btn ${activeTab === "report" ? "tab-btn--active" : ""}`}
              onClick={() => setActiveTab("report")}
            >
              <span className="tab-icon">📄</span> Master Report
            </button>
            <button
              className={`tab-btn ${activeTab === "papers" ? "tab-btn--active" : ""}`}
              onClick={() => setActiveTab("papers")}
            >
              <span className="tab-icon"><IconBook /></span> Papers ({papers.length})
            </button>
            <button
              className={`tab-btn ${activeTab === "analyses" ? "tab-btn--active" : ""}`}
              onClick={() => setActiveTab("analyses")}
            >
              <span className="tab-icon"><IconFlask /></span> Analyses ({analyses.length})
            </button>
          </div>

          {/* Tab: Report */}
          {activeTab === "report" && (
            <div className="tab-panel">
              <div className="tab-panel__toolbar">
                <button className="action-btn" onClick={handleDownload}>
                  <IconDownload /> Download .md
                </button>
              </div>
              <div className="report-body">
                {report ? (
                  <pre className="report-pre">{report}</pre>
                ) : (
                  <p className="empty-state">No report generated yet.</p>
                )}
              </div>
            </div>
          )}

          {/* Tab: Papers */}
          {activeTab === "papers" && (
            <div className="tab-panel">
              <div className="papers-grid">
                {papers.map((p) => (
                  <PaperCard
                    key={p.arxiv_id}
                    paper={p}
                    onSelect={handlePaperSelect}
                    selected={selectedPaper === p.arxiv_id}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Tab: Analyses */}
          {activeTab === "analyses" && (
            <div className="tab-panel">
              {analyses.length ? (
                analyses.map((a) => <AnalysisCard key={a.arxiv_id} analysis={a} />)
              ) : (
                <p className="empty-state">No structured analyses yet.</p>
              )}
            </div>
          )}
        </>
      )}

      {/* ── Empty / Initial state ── */}
      {!is_loading && !papers.length && !report && (
        <div className="rp-empty">
          <div className="rp-empty__icon">🔬</div>
          <div className="rp-empty__title">Ready to Research</div>
          <div className="rp-empty__subtitle">Enter a topic above and hit Start Deep Research</div>
        </div>
      )}
    </div>
  )
}

export default withStreamlitConnection(ResearchPanel)
