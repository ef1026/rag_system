import { Suspense } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { QuizPracticePage } from "@/features/quiz/QuizPracticePage";

export default function QuizPage() {
  return (
    <AppShell>
      <Suspense fallback={<p className="empty-state">正在加载做题页面...</p>}>
        <QuizPracticePage />
      </Suspense>
    </AppShell>
  );
}
