
import React, { useEffect, useRef, useState } from 'react';
import {
  BookOpen, UploadCloud, FileText, Send, Plus, RefreshCw,
  MessageSquareText, ChevronDown, CheckCircle2, AlertCircle,
  LoaderCircle, PanelLeftClose, PanelLeftOpen, Sparkles, RotateCcw,
  Trash2
} from 'lucide-react';

const API = '/api';

const KEYS = {
  messages: 'atlas_messages',
  sessions: 'atlas_sessions',
  selected: 'atlas_selected_document',
  jobs: 'atlas_indexing_jobs'
};

function loadSaved(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : JSON.parse(value);
  } catch {
    return fallback;
  }
}

function saveData(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn('Atlas storage error:', error);
  }
}

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(API + path, options);
  } catch {
    throw new Error('Cannot reach FastAPI. Start the backend on port 8000.');
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      typeof data.detail === 'string'
        ? data.detail
        : `Request failed (${response.status})`
    );
  }

  return data;
}

function uniqueName(file) {
  const base = file.name
    .replace(/\.pdf$/i, '')
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 40) || 'document';

  return `${base}_${Date.now().toString(36)}`;
}

function TextWithCitations({ text = '', sources = [] }) {
  return (
    <>
      {String(text).split(/(\[S\d+\])/g).map((part, index) => {
        const source = sources.find(
          s => `[${s.citation_id}]` === part
        );

        return source ? (
          <span
            key={index}
            className="citation"
            title={`${source.source}, page ${source.page ?? '?'}`}
          >
            {part}
          </span>
        ) : (
          <React.Fragment key={index}>{part}</React.Fragment>
        );
      })}
    </>
  );
}

export default function App() {
  const [collections, setCollections] = useState([]);
  const [selected, setSelected] = useState(
    () => loadSaved(KEYS.selected, '')
  );
  const [messages, setMessages] = useState(
    () => loadSaved(KEYS.messages, {})
  );
  const [jobs, setJobs] = useState(
    () => loadSaved(KEYS.jobs, {})
  );

  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [busyCollection, setBusyCollection] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadName, setUploadName] = useState('');
  const [error, setError] = useState('');
  const [sidebar, setSidebar] = useState(true);
  const [deletingCollection, setDeletingCollection] = useState('');

  const fileRef = useRef(null);
  const bottom = useRef(null);
  const input = useRef(null);
  const busyRef = useRef(false);
  const uploadRef = useRef(false);
  const jobsRef = useRef(jobs);
  const sessionsRef = useRef(loadSaved(KEYS.sessions, {}));

  const current = messages[selected] || [];
  const chosen = collections.find(c => c.name === selected);
  const jobList = Object.values(jobs);
  const activeJobs = jobList.filter(
    job => job.status === 'queued' || job.status === 'processing'
  );

  useEffect(() => saveData(KEYS.messages, messages), [messages]);
  useEffect(() => saveData(KEYS.selected, selected), [selected]);

  useEffect(() => {
    jobsRef.current = jobs;
    saveData(KEYS.jobs, jobs);
  }, [jobs]);

  useEffect(() => {
    loadCollections();
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' });
  }, [current.length, busy, selected]);

  function getSessionId(collection) {
    if (!sessionsRef.current[collection]) {
      sessionsRef.current[collection] = crypto.randomUUID();
      saveData(KEYS.sessions, sessionsRef.current);
    }

    return sessionsRef.current[collection];
  }

  async function loadCollections(preferred = null) {
    try {
      const data = await request('/collections');
      const list = data.collections || [];

      setCollections(list);

      setSelected(previous => {
        if (preferred && list.some(c => c.name === preferred)) {
          return preferred;
        }

        if (previous && list.some(c => c.name === previous)) {
          return previous;
        }

        return list[0]?.name || '';
      });

      return list;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  // Poll active background indexing jobs every two seconds.
  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function pollJobs() {
      const pending = Object.values(jobsRef.current).filter(
        job => job.status === 'queued' || job.status === 'processing'
      );

      if (!pending.length) return;

      const updates = await Promise.all(
        pending.map(async job => {
          try {
            const result = await request(
              `/jobs/${encodeURIComponent(job.job_id)}`
            );
            return { previous: job, result };
          } catch (err) {
            return { previous: job, error: err.message };
          }
        })
      );

      if (cancelled) return;

      let completedCollection = null;
      let shouldRefresh = false;

      const next = { ...jobsRef.current };

      for (const update of updates) {
        const id = update.previous.job_id;
        const existing = next[id];

        if (!existing) continue;

        if (
          existing.status !== 'queued' &&
          existing.status !== 'processing'
        ) continue;

        if (update.error) {
          next[id] = {
            ...existing,
            status: 'failed',
            error: update.error
          };
          continue;
        }

        next[id] = {
          ...existing,
          ...update.result
        };

        if (update.result.status === 'completed') {
          shouldRefresh = true;
          completedCollection = update.result.collection;
        }
      }

      jobsRef.current = next;
      setJobs(next);

      if (shouldRefresh) {
        await loadCollections(completedCollection);
      }
    }

    async function tick() {
      await pollJobs();

      if (!cancelled) {
        timer = setTimeout(tick, 2000);
      }
    }

    tick();

    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, []);

  async function upload(file) {
    if (!file || uploadRef.current) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file.');
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setError('PDF must be smaller than 25 MB.');
      return;
    }

    uploadRef.current = true;
    setUploading(true);
    setUploadName(file.name);
    setError('');

    const collection = uniqueName(file);

    try {
      const form = new FormData();
      form.append('collection', collection);
      form.append('file', file);

      const data = await request('/upload', {
        method: 'POST',
        body: form
      });

      if (!data.job_id) {
        throw new Error(
          'Upload response has no job ID. Check that the updated backend is running.'
        );
      }

      const job = {
        job_id: data.job_id,
        collection,
        filename: file.name,
        status: data.status || 'queued',
        progress: 0,
        error: null
      };

      jobsRef.current = {
        ...jobsRef.current,
        [job.job_id]: job
      };

      setJobs({ ...jobsRef.current });
    } catch (err) {
      setError(err.message);
    } finally {
      uploadRef.current = false;
      setUploading(false);
      setUploadName('');

      if (fileRef.current) fileRef.current.value = '';
    }
  }

  async function ask(event) {
    event?.preventDefault();

    const q = question.trim();
    const collection = selected;

    if (!q || !collection || busyRef.current || deletingCollection) return;

    busyRef.current = true;
    setBusy(true);
    setBusyCollection(collection);
    setQuestion('');
    setError('');

    const sessionId = getSessionId(collection);

    setMessages(previous => ({
      ...previous,
      [collection]: [
        ...(previous[collection] || []),
        { role: 'user', text: q }
      ]
    }));

    try {
      const data = await request('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          collection,
          session_id: sessionId,
          top_k: 5
        })
      });

      setMessages(previous => ({
        ...previous,
        [collection]: [
          ...(previous[collection] || []),
          {
            role: 'assistant',
            text: data.answer,
            sources: data.sources || [],
            route: data.route,
            rewritten_question: data.rewritten_question
          }
        ]
      }));
    } catch (err) {
      setMessages(previous => ({
        ...previous,
        [collection]: [
          ...(previous[collection] || []),
          { role: 'error', text: err.message }
        ]
      }));
    } finally {
      busyRef.current = false;
      setBusy(false);
      setBusyCollection('');
      input.current?.focus();
    }
  }

  async function newChat() {
    if (!selected || busyRef.current || deletingCollection) return;

    const collection = selected;
    const oldSession = sessionsRef.current[collection];

    if (oldSession) {
      try {
        await request('/conversation/clear', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            collection,
            session_id: oldSession
          })
        });
      } catch (err) {
        setError(`Failed to clear conversation: ${err.message}`);
        return;
      }
    }

    sessionsRef.current[collection] = crypto.randomUUID();
    saveData(KEYS.sessions, sessionsRef.current);

    setMessages(previous => ({
      ...previous,
      [collection]: []
    }));

    setQuestion('');
    setError('');
    input.current?.focus();
  }

  // Delete indexed document and its local conversation data.
  async function deleteDocument(collection) {
    if (deletingCollection || busyRef.current) return;

    const document = collections.find(c => c.name === collection);
    const label = document?.filename || collection;

    if (!window.confirm(
      `Permanently delete "${label}" and its conversation history? This cannot be undone.`
    )) {
      return;
    }

    setDeletingCollection(collection);
    setError('');

    try {
      await request(
        `/collections/${encodeURIComponent(collection)}`,
        { method: 'DELETE' }
      );

      // Remove the deleted document's messages.
      setMessages(previous => {
        const next = { ...previous };
        delete next[collection];
        saveData(KEYS.messages, next);
        return next;
      });

      // Remove the deleted document's session.
      delete sessionsRef.current[collection];
      saveData(KEYS.sessions, sessionsRef.current);

      // Remove completed jobs associated with this document.
      const nextJobs = { ...jobsRef.current };
      for (const [id, job] of Object.entries(nextJobs)) {
        if (job.collection === collection) {
          delete nextJobs[id];
        }
      }

      jobsRef.current = nextJobs;
      setJobs(nextJobs);

      // Refresh the sidebar and select another available PDF.
      const remaining = collections.filter(
        c => c.name !== collection
      );

      setCollections(remaining);

      if (selected === collection) {
        setSelected(remaining[0]?.name || '');
        setQuestion('');
      }

      await loadCollections();
    } catch (err) {
      setError(`Failed to delete document: ${err.message}`);
    } finally {
      setDeletingCollection('');
    }
  }

  return (
    <div className="app">
      <aside className={`sidebar ${sidebar ? '' : 'collapsed'}`}>
        <div className="brand">
          <div className="brand-icon">
            <BookOpen size={20} />
          </div>
          <span>atlas<span className="brand-dot">.</span></span>
        </div>

        <div className="sidebar-heading">WORKSPACE</div>

        <button
          className="upload-button"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
        >
          <Plus size={17} /> Add document
        </button>

        <input
          ref={fileRef}
          type="file"
          accept=".pdf,application/pdf"
          hidden
          onChange={e => upload(e.target.files?.[0])}
        />

        <div className="section-label">
          <span>YOUR DOCUMENTS</span>
          <button
            title="Refresh collections"
            onClick={() => loadCollections()}
          >
            <RefreshCw size={14} />
          </button>
        </div>

        <div className="document-list">
          {collections.map(c => (
            <div
              key={c.name}
              style={{
                display: 'flex',
                alignItems: 'center',
                position: 'relative'
              }}
            >
              <button
                className={`document ${selected === c.name ? 'active' : ''}`}
                style={{ flex: 1, minWidth: 0, paddingRight: 42 }}
                onClick={() => {
                  setSelected(c.name);
                  setError('');
                }}
              >
                <span className="doc-icon">
                  <FileText size={18} />
                </span>
                <span className="doc-copy">
                  <strong title={c.filename}>
                    {c.filename || c.name}
                  </strong>
                  <small>
                    {c.pages} {c.pages === 1 ? 'page' : 'pages'} · Indexed
                  </small>
                </span>
                {selected === c.name && (
                  <span className="selected-dot" />
                )}
              </button>

              <button
                type="button"
                title={`Delete ${c.filename || c.name}`}
                aria-label={`Delete ${c.filename || c.name}`}
                disabled={Boolean(deletingCollection) || busy}
                onClick={() => deleteDocument(c.name)}
                style={{
                  position: 'absolute',
                  right: 9,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: 6,
                  border: 'none',
                  background: 'transparent',
                  color: '#8b6b65',
                  cursor: deletingCollection || busy ? 'not-allowed' : 'pointer',
                  zIndex: 2
                }}
              >
                {deletingCollection === c.name
                  ? <LoaderCircle size={16} className="spin" />
                  : <Trash2 size={16} />}
              </button>
            </div>
          ))}

          {activeJobs.map(job => (
            <div
              key={job.job_id}
              className="document"
              style={{ cursor: 'default', opacity: 0.8 }}
            >
              <span className="doc-icon">
                <LoaderCircle size={18} className="spin" />
              </span>
              <span className="doc-copy">
                <strong title={job.filename}>
                  {job.filename}
                </strong>
                <small>
                  {job.status === 'queued'
                    ? 'Queued for indexing'
                    : 'Indexing in progress'}
                </small>
              </span>
            </div>
          ))}

          {!collections.length && !activeJobs.length && (
            <p className="empty-list">
              No documents yet. Upload a PDF to begin.
            </p>
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="status-dot" /> Local RAG workspace
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="top-left">
            <button
              className="icon-button toggle"
              onClick={() => setSidebar(!sidebar)}
              title="Toggle sidebar"
            >
              {sidebar
                ? <PanelLeftClose size={19} />
                : <PanelLeftOpen size={19} />}
            </button>
            <span className="crumb">Workspace</span>
            <span className="slash">/</span>
            <strong>{chosen?.filename || 'Document chat'}</strong>
          </div>

          <div className="top-right">
            {selected && (
              <button
                className="icon-button"
                onClick={newChat}
                disabled={busy || Boolean(deletingCollection)}
                title="New chat"
                aria-label="New chat"
              >
                <RotateCcw size={17} />
              </button>
            )}
            <span className="engine-status">
              <span /> Engine connected via FastAPI
            </span>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <AlertCircle size={17} />
            <span>{error}</span>
            <button onClick={() => setError('')}>Dismiss</button>
          </div>
        )}

        {uploading && (
          <div className="upload-banner">
            <LoaderCircle className="spin" size={18} />
            <span>
              Uploading <strong>{uploadName}</strong>…
            </span>
          </div>
        )}

        {activeJobs.map(job => (
          <div className="upload-banner" key={job.job_id}>
            <LoaderCircle className="spin" size={18} />
            <span>
              <strong>{job.filename}</strong> —{' '}
              {job.status === 'queued'
                ? 'Waiting for indexing worker…'
                : 'Building RAPTOR tree and ColBERT index…'}
            </span>
          </div>
        ))}

        {jobList.filter(job => job.status === 'failed').map(job => (
          <div className="error-banner" key={job.job_id}>
            <AlertCircle size={17} />
            <span>
              Indexing failed for {job.filename}: {job.error || 'Unknown error'}
            </span>
            <button
              onClick={() => {
                const next = { ...jobsRef.current };
                delete next[job.job_id];
                jobsRef.current = next;
                setJobs(next);
              }}
            >
              Dismiss
            </button>
          </div>
        ))}

        <div className="chat-scroll">
          <div className="chat-inner">
            {!current.length ? (
              <div className="welcome">
                <div className="welcome-icon">
                  <Sparkles size={27} />
                </div>
                <div className="eyebrow">
                  YOUR DOCUMENT INTELLIGENCE WORKSPACE
                </div>
                <h1>
                  Ask your documents<br />
                  <em>anything.</em>
                </h1>
                <p>
                  Upload a PDF, ask questions in plain language, and get
                  answers grounded in your document with page-level citations.
                </p>

                {chosen ? (
                  <>
                    <div className="selected-card">
                      <FileText size={20} />
                      <div>
                        <strong>{chosen.filename}</strong>
                        <span>
                          {chosen.pages} pages · Ready to explore
                        </span>
                      </div>
                      <CheckCircle2 size={19} className="check" />
                    </div>
                    <div className="prompts">
                      {[
                        'Summarize this document',
                        'What are the key findings?',
                        'Explain the main concepts'
                      ].map(prompt => (
                        <button
                          key={prompt}
                          onClick={() => {
                            setQuestion(prompt);
                            input.current?.focus();
                          }}
                        >
                          {prompt}<span>↗</span>
                        </button>
                      ))}
                    </div>
                  </>
                ) : activeJobs.length ? (
                  <p>
                    Your PDF is being indexed. It will appear here when ready.
                  </p>
                ) : (
                  <button
                    className="welcome-upload"
                    onClick={() => fileRef.current?.click()}
                  >
                    <UploadCloud size={18} /> Upload your first PDF
                  </button>
                )}
              </div>
            ) : (
              <div className="thread">
                {current.map((message, index) => (
                  <div
                    key={index}
                    className={`message ${message.role}`}
                  >
                    <div className="avatar">
                      {message.role === 'user'
                        ? 'You'
                        : message.role === 'error'
                        ? <AlertCircle size={17} />
                        : <Sparkles size={17} />}
                    </div>

                    <div className="message-body">
                      <div className="speaker">
                        {message.role === 'user'
                          ? 'You'
                          : message.role === 'error'
                          ? 'Error'
                          : 'Atlas'}
                      </div>

                      <div className="message-text">
                        <TextWithCitations
                          text={message.text}
                          sources={message.sources}
                        />
                      </div>

                      {message.role === 'assistant' &&
                        message.sources?.length > 0 && (
                        <details className="sources">
                          <summary>
                            <BookOpen size={14} />
                            {message.sources.length} source passages
                            <ChevronDown size={14} />
                          </summary>
                          <div className="source-list">
                            {message.sources.map(source => (
                              <div
                                key={source.citation_id}
                                className="source"
                              >
                                <span className="source-id">
                                  [{source.citation_id}]
                                </span>
                                <span>
                                  {source.source} · Page{' '}
                                  {source.page ?? 'unknown'}
                                </span>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  </div>
                ))}

                {busy && busyCollection === selected && (
                  <div className="message assistant">
                    <div className="avatar">
                      <Sparkles size={17} />
                    </div>
                    <div className="message-body">
                      <div className="speaker">Atlas</div>
                      <div className="thinking">
                        <span /><span /><span />
                        Retrieving evidence and generating an answer…
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            <div ref={bottom} />
          </div>
        </div>

        <div className="composer-wrap">
          <form className="composer" onSubmit={ask}>
            <textarea
              ref={input}
              rows={1}
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  ask();
                }
              }}
              disabled={!selected || busy || Boolean(deletingCollection)}
              placeholder={
                selected
                  ? 'Ask a question about your document...'
                  : 'Upload or select a document to start...'
              }
            />
            <div className="composer-bottom">
              <span>
                <MessageSquareText size={14} />
                Answers grounded in your document
              </span>
              <button
                className="send"
                type="submit"
                disabled={
                  !question.trim() ||
                  !selected ||
                  busy ||
                  Boolean(deletingCollection)
                }
                title="Send question"
              >
                <Send size={17} />
              </button>
            </div>
          </form>
          <div className="footnote">
            Atlas uses local Qwen3, RAPTOR and ColBERT.
            Verify important information against the cited document.
          </div>
        </div>
      </main>
    </div>
  );
}
