type ProfilePromptPreviewProps = {
  promptContext: string;
  isLoading: boolean;
};

export function ProfilePromptPreview({
  promptContext,
  isLoading,
}: ProfilePromptPreviewProps) {
  return (
    <section className="panel profile-preview">
      <div className="panel-heading">
        <div>
          <p className="section-label">Prompt Preview</p>
          <h2>Profile context</h2>
        </div>
      </div>
      <pre className="prompt-preview-text">
        {isLoading ? "Loading profile context..." : promptContext}
      </pre>
    </section>
  );
}
