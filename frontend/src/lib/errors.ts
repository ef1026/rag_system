export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function parseApiError(response: Response): Promise<ApiError> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return new ApiError(payload.detail || response.statusText, response.status);
  } catch {
    return new ApiError(response.statusText, response.status);
  }
}
