export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function sourceLabel(type: string): string {
  const labels: Record<string, string> = {
    text: "正文",
    equation: "公式",
    table: "表格",
  };
  return labels[type] || type;
}
