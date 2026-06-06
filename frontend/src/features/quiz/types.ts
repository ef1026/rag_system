export type QuizChoice = {
  id: string;
  text: string;
};

export type QuizQuestion = {
  id: string;
  prompt: string;
  choices: QuizChoice[];
  correct_choice_id: string;
  explanation: string;
  source_message_ids: string[];
};

export type QuizSession = {
  id: string;
  profile_id: string;
  conversation_id: string;
  title: string;
  questions: QuizQuestion[];
  created_at: string;
};

export type QuizSubmitAnswer = {
  question_id: string;
  selected_choice_id: string;
};

export type QuizQuestionResult = {
  question: QuizQuestion;
  selected_choice_id?: string | null;
  is_correct: boolean;
  correct_choice_id: string;
  explanation: string;
};

export type QuizSubmitResponse = {
  session_id: string;
  correct_count: number;
  total_count: number;
  results: QuizQuestionResult[];
};

export type WrongQuestion = {
  id: string;
  profile_id: string;
  quiz_session_id: string;
  conversation_id: string;
  question_id: string;
  prompt: string;
  choices: QuizChoice[];
  selected_choice_id: string;
  correct_choice_id: string;
  explanation: string;
  source_message_ids: string[];
  created_at: string;
  reviewed_at?: string | null;
};
