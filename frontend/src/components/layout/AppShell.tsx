import Link from "next/link";
import type { ReactNode } from "react";

type AppShellProps = {
  children: ReactNode;
  status?: ReactNode;
};

export function AppShell({ children, status }: AppShellProps) {
  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <p className="eyebrow">RAG 助教</p>
          <h1>文档解析与知识问答工作台</h1>
        </div>
        <div className="topbar-meta">
          <nav className="app-nav" aria-label="主导航">
            <Link href="/">问答</Link>
            <Link href="/quiz">小测</Link>
            <Link href="/wrong-book">错题</Link>
            <Link href="/profile">画像</Link>
            <Link href="/library">资料库</Link>
          </nav>
          {status ? <div className="runtime-strip">{status}</div> : null}
          <div className="status-pill">后端 API + 前端界面</div>
        </div>
      </header>
      {children}
    </main>
  );
}
