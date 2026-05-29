export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export async function parseApiError(response: Response): Promise<ApiError> {
  try {
    const payload = (await response.json()) as {
      detail?: string;
      error?: string;
      message?: string;
    };
    return new ApiError(
      payload.message || payload.detail || response.statusText,
      response.status,
      payload.error,
    );
  } catch {
    return new ApiError(response.statusText, response.status);
  }
}
