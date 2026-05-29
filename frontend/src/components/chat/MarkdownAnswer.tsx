"use client";

import { Component, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { RelatedImageInlineCard } from "@/components/chat/RelatedImages";
import type { RelatedImage } from "@/types/chat";

type MarkdownAnswerProps = {
  content: string;
  relatedImages?: RelatedImage[];
};

type AnswerBlock =
  | {
      type: "text";
      content: string;
    }
  | {
      type: "image";
      imageId: string;
    };

type MarkdownErrorBoundaryProps = {
  children: ReactNode;
  fallback: string;
};

type MarkdownErrorBoundaryState = {
  hasError: boolean;
};

class MarkdownErrorBoundary extends Component<
  MarkdownErrorBoundaryProps,
  MarkdownErrorBoundaryState
> {
  state: MarkdownErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): MarkdownErrorBoundaryState {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return <p className="markdown-answer-fallback">{this.props.fallback}</p>;
    }

    return this.props.children;
  }
}

export function extractInlineImageIds(content: string) {
  const imageIds: string[] = [];
  const seen = new Set<string>();
  const pattern = /\[\[image:\s*([A-Za-z0-9._:-]+)\s*\]\]/g;
  for (const match of content.matchAll(pattern)) {
    const imageId = match[1]?.trim();
    if (imageId && !seen.has(imageId)) {
      imageIds.push(imageId);
      seen.add(imageId);
    }
  }
  return imageIds;
}

function parseAnswerBlocks(content: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  // Inline image tokens are backend-issued references mapped to API-hosted assets.
  const pattern = /\[\[image:\s*([A-Za-z0-9._:-]+)\s*\]\]/g;
  let lastIndex = 0;

  for (const match of content.matchAll(pattern)) {
    const index = match.index ?? 0;
    const text = content.slice(lastIndex, index);
    if (text) {
      blocks.push({ type: "text", content: text });
    }
    const imageId = match[1]?.trim();
    if (imageId) {
      blocks.push({ type: "image", imageId });
    }
    lastIndex = index + match[0].length;
  }

  const tail = content.slice(lastIndex);
  if (tail) {
    blocks.push({ type: "text", content: tail });
  }

  return blocks.length ? blocks : [{ type: "text", content }];
}

export function MarkdownAnswer({ content, relatedImages }: MarkdownAnswerProps) {
  const imagesById = new Map(
    relatedImages?.map((image) => [image.imageId, image]) ?? [],
  );
  const blocks = parseAnswerBlocks(content);

  return (
    <MarkdownErrorBoundary fallback={content}>
      <div className="markdown-answer">
        {blocks.map((block, index) => {
          if (block.type === "image") {
            const image = imagesById.get(block.imageId);
            if (!image) {
              return (
                <p className="inline-image-warning" key={`${block.imageId}-${index}`}>
                  图片不可用：{block.imageId}
                </p>
              );
            }
            return (
              <RelatedImageInlineCard
                image={image}
                key={`${image.documentId}-${image.imageId}-${index}`}
              />
            );
          }

          return (
            <ReactMarkdown
              key={`text-${index}`}
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[
                [rehypeKatex, { strict: false, throwOnError: false }],
              ]}
            >
              {block.content}
            </ReactMarkdown>
          );
        })}
      </div>
    </MarkdownErrorBoundary>
  );
}
