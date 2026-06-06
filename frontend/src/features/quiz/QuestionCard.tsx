"use client";

import { MarkdownAnswer } from "@/components/chat/MarkdownAnswer";
import type { QuizChoice, QuizQuestion, QuizQuestionResult } from "./types";

type QuestionDisplay = Pick<
  QuizQuestion,
  "id" | "prompt" | "choices" | "correct_choice_id" | "explanation" | "source_message_ids"
>;

type QuestionCardProps = {
  question: QuestionDisplay;
  index: number;
  selectedChoiceId?: string;
  result?: QuizQuestionResult;
  showAnswer?: boolean;
  onSelect?: (choiceId: string) => void;
};

export function QuestionCard({
  question,
  index,
  selectedChoiceId,
  result,
  showAnswer = false,
  onSelect,
}: QuestionCardProps) {
  const isAnswered = Boolean(result || showAnswer);
  const correctChoiceId = result?.correct_choice_id || question.correct_choice_id;
  const explanation = result?.explanation || question.explanation;

  return (
    <article className="quiz-question">
      <div className="quiz-question-prompt">
        <span className="quiz-question-number">{index + 1}</span>
        <MarkdownAnswer content={question.prompt} />
      </div>

      <SourceReferences sourceIds={question.source_message_ids} />

      <div className="quiz-choice-list">
        {question.choices.map((choice) => (
          <ChoiceButton
            key={choice.id}
            choice={choice}
            correctChoiceId={correctChoiceId}
            isAnswered={isAnswered}
            selectedChoiceId={selectedChoiceId}
            onSelect={onSelect}
          />
        ))}
      </div>

      {isAnswered ? (
        <div className="quiz-explanation">
          <strong>解析</strong>
          <MarkdownAnswer content={explanation || "暂无解析。"} />
        </div>
      ) : null}
    </article>
  );
}

function ChoiceButton({
  choice,
  correctChoiceId,
  isAnswered,
  selectedChoiceId,
  onSelect,
}: {
  choice: QuizChoice;
  correctChoiceId: string;
  isAnswered: boolean;
  selectedChoiceId?: string;
  onSelect?: (choiceId: string) => void;
}) {
  const isSelected = selectedChoiceId === choice.id;
  const isCorrect = isAnswered && choice.id === correctChoiceId;
  const isWrongSelection = isAnswered && isSelected && !isCorrect;

  return (
    <button
      type="button"
      className={[
        "quiz-choice",
        isSelected ? "selected" : "",
        isCorrect ? "correct" : "",
        isWrongSelection ? "incorrect" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      disabled={isAnswered || !onSelect}
      onClick={() => onSelect?.(choice.id)}
    >
      <span>{choice.id}</span>
      <em>{choice.text}</em>
    </button>
  );
}

function SourceReferences({ sourceIds }: { sourceIds: string[] }) {
  const labels = uniqueSourceLabels(sourceIds);
  if (!labels.length) return null;

  return (
    <div className="quiz-source-list" aria-label="题目引用来源">
      {labels.map((label) => (
        <span key={label}>{label}</span>
      ))}
    </div>
  );
}

function uniqueSourceLabels(sourceIds: string[]) {
  return Array.from(new Set(sourceIds.map(sourceLabelForId)));
}

function sourceLabelForId(sourceId: string) {
  if (sourceId.startsWith("wrong_")) return "引用错题记录";
  if (sourceId.startsWith("quiz_")) return "引用历史小测";
  if (sourceId.startsWith("learn_")) return "引用问答记录";
  if (sourceId.startsWith("mem_")) return "引用学习记忆";
  return "引用学习材料";
}

export function choiceText(
  question: Pick<QuestionDisplay, "choices">,
  choiceId: string,
) {
  return question.choices.find((choice) => choice.id === choiceId)?.text || choiceId;
}
