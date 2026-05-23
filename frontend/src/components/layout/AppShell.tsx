import type { ReactNode } from "react";

type AppShellProps = {
  children: ReactNode;
};

export function AppShell({ children }: AppShellProps) {
  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">RAG Teaching Assistant</p>
          <h1>文档解析与知识问答工作台</h1>
        </div>
        <div className="status-pill">FastAPI + Next.js</div>
      </header>
      {children}
    </main>
  );
}
