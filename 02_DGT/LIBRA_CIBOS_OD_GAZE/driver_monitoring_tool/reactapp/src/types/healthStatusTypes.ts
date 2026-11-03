export type HealthStatus = 
    "disconnected" |
    "connecting" |
    "connected" |
    "subscribed" |
    "invalid_data" |
    "error" |
    "retrying";

export const STATUS_MESSAGE: Record<HealthStatus, string> = {
  "disconnected": "Disconnected",
  "connecting": "Connecting",
  "connected": "Connected",
  "subscribed": "Subscribed",
  "invalid_data": "Invalid Data",
  "error": "Error",
  "retrying": "Trying to reconnect..."
};

export interface HealthStatusLog {
    subscriber: {
        status: HealthStatus,
        last_error: string | null;
        seconds_since_last_message: number | null;
        seconds_since_last_valid_message: number | null;
        seconds_since_last_invalid_message: number | null;
        healthy: boolean;
    } | null;
    publisher: {
        status: HealthStatus,
        last_error: string | null;
        seconds_since_last_publish: number | null;
        healthy: boolean;
    } | null;
}