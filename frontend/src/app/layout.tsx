import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Teaching Assistant",
  description: "Document parsing and RAG question answering workspace.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
