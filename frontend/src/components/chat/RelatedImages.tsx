"use client";

import { useEffect, useState } from "react";
import { getApiUrl } from "@/lib/api";
import type { RelatedImage } from "@/types/chat";

type RelatedImagesProps = {
  images?: RelatedImage[];
};

type RelatedImageInlineCardProps = {
  image: RelatedImage;
};

type RelatedImagePreviewProps = {
  image: RelatedImage;
  className: string;
};

export function RelatedImages({ images }: RelatedImagesProps) {
  const { selectedImage, setSelectedImage } = useImageLightbox();

  if (!images?.length) return null;

  return (
    <section className="related-images" aria-label="相关图片">
      <h3>相关图片</h3>
      <div className="related-image-grid">
        {images.map((image) => (
          <button
            type="button"
            className="related-image-card"
            key={`${image.documentId}-${image.imageId}`}
            onClick={() => setSelectedImage(image)}
          >
            <ImagePreview image={image} className="related-image-thumbnail" />
            <span className="related-image-caption">
              {image.caption || "未命名图片"}
            </span>
            <span className="related-image-meta">
              {image.documentName || image.documentId}
              {typeof image.page === "number" ? ` · 第 ${image.page} 页` : ""}
            </span>
          </button>
        ))}
      </div>

      <ImageLightbox image={selectedImage} onClose={() => setSelectedImage(null)} />
    </section>
  );
}

export function RelatedImageInlineCard({ image }: RelatedImageInlineCardProps) {
  const { selectedImage, setSelectedImage } = useImageLightbox();

  return (
    <>
      <button
        type="button"
        className="inline-image-card"
        onClick={() => setSelectedImage(image)}
      >
        <ImagePreview image={image} className="inline-image-preview" />
        <span className="inline-image-body">
          <span className="inline-image-caption">
            {image.caption || "未命名图片"}
          </span>
          <span className="inline-image-meta">
            来源：{image.documentName || image.documentId}
            {typeof image.page === "number" ? ` · 第 ${image.page} 页` : ""}
          </span>
        </span>
      </button>
      <ImageLightbox image={selectedImage} onClose={() => setSelectedImage(null)} />
    </>
  );
}

function useImageLightbox() {
  const [selectedImage, setSelectedImage] = useState<RelatedImage | null>(null);

  useEffect(() => {
    if (!selectedImage) return;

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSelectedImage(null);
      }
    }

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [selectedImage]);

  return { selectedImage, setSelectedImage };
}

function ImageLightbox({
  image,
  onClose,
}: {
  image: RelatedImage | null;
  onClose: () => void;
}) {
  if (!image) return null;

  return (
    <div
      className="image-lightbox"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        className="image-lightbox-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="图片预览"
      >
        <button
          type="button"
          className="image-lightbox-close"
          aria-label="关闭图片预览"
          onClick={onClose}
        >
          ×
        </button>
        <ImagePreview image={image} className="image-lightbox-image" />
        <div className="image-lightbox-meta">
          <strong>{image.caption || "未命名图片"}</strong>
          <span>
            来源：{image.documentName || image.documentId}
            {typeof image.page === "number" ? ` · 第 ${image.page} 页` : ""}
          </span>
        </div>
      </div>
    </div>
  );
}

function ImagePreview({ image, className }: RelatedImagePreviewProps) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <span className={`${className} related-image-placeholder`}>
        图片加载失败
      </span>
    );
  }

  return (
    // Backend image URLs are runtime API assets, so Next Image cannot know the host at build time.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      className={className}
      src={getApiUrl(image.url)}
      alt={image.caption || image.documentName || image.documentId}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}
