export interface JanusMedia {
  mid: string;
  type: "video" | "audio" | "data";
  codec: string;
}

export interface JanusMountpoint {
  id: number;
  description: string;
  type: string;
  media: JanusMedia[];
}

export interface PluginHandle {
  send: (options: { 
    message: Record<string, unknown>; 
    jsep?: unknown;
    success?: (result: unknown) => void;
    error?: (err: unknown) => void;
   }) => void;
  createAnswer: (options: {
    jsep: unknown;
    video?: { recv?: boolean; add?: boolean };
    media?: { audioSend: boolean; videoSend: boolean };
    success: (jsep: unknown) => void;
    error: (e: unknown) => void;
  }) => void;
  detach: () => void;
}