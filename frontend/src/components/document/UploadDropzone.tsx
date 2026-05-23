"use client";

import { useRef } from "react";
import { Button } from "@/components/ui/Button";

type UploadDropzoneProps = {
  disabled?: boolean;
  onUpload: (file: File) => void;
};

export function UploadDropzone({ disabled, onUpload }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <section className="upload-zone">
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onUpload(file);
          event.currentTarget.value = "";
        }}
      />
      <div>
        <p className="section-label">PDF 教材</p>
        <h2>上传或选择已有文档</h2>
        <p className="muted">解析和索引都在后端完成，前端只保存当前操作状态。</p>
      </div>
      <Button
        type="button"
        variant="primary"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        上传 PDF
      </Button>
    </section>
  );
}
