type TagEditorProps = {
  value: string;
  onChange: (value: string) => void;
};

export function TagEditor({ value, onChange }: TagEditorProps) {
  const tags = parseTags(value);

  return (
    <label className="form-field">
      <span>标签</span>
      <input
        value={value}
        placeholder="数学, 讲义, 复习"
        onChange={(event) => onChange(event.target.value)}
      />
      {tags.length ? (
        <div className="tag-row">
          {tags.map((tag) => (
            <span className="tag-chip" key={tag}>
              {tag}
            </span>
          ))}
        </div>
      ) : null}
    </label>
  );
}

export function parseTags(value: string) {
  const seen = new Set<string>();
  const tags: string[] = [];
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .forEach((item) => {
      const key = item.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        tags.push(item);
      }
    });
  return tags;
}
