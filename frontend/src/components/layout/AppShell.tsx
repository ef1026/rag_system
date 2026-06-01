import type { ReactNode } from "react";
import Link from "next/link";

type AppShellProps = {
  children: ReactNode;
  status?: ReactNode;
};

export function AppShell({ children, status }: AppShellProps) {
  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <p className="eyebrow">RAG Teaching Assistant</p>
          <h1>文档解析与知识问答工作台</h1>
        </div>
        <div className="topbar-meta">
          <nav className="app-nav" aria-label="Primary navigation">
            <Link href="/">Chat</Link>
            <Link href="/profile">Profile</Link>
            <Link href="/library">Library</Link>
          </nav>
          {status ? <div className="runtime-strip">{status}</div> : null}
          <div className="status-pill">FastAPI + Next.js</div>
        </div>
      </header>
      {children}
    </main>
  );
}
