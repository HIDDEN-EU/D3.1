import { useEffect, useState } from "react";
import { dataManager } from "../helpers/dataManager";
import type { HealthStatus, HealthStatusLog } from "../types/healthStatusTypes";

interface HealthCheckOptions {
    interval?: number;       //time between checks while healthy (ms)
    retryInterval?: number;  //time between retries after a failure (ms)
    timeout?: number;        //abort a request that takes longer than this (ms)
}

export function useHealthCheck(
    endpoint: string,
    { interval = 15000, retryInterval = 4000, timeout = 10000 }: HealthCheckOptions = {}
) {
    const [connected, setConnected] = useState<boolean>(false);
    const [status, setStatus] = useState<{publisher: HealthStatus; subscriber: HealthStatus}>(
        {publisher: "connecting", subscriber: "connecting"}
    );
    const [errors, setErrors] = useState<{publisher: string | null; subscriber: string | null}>(
        {publisher: null, subscriber: null}
    );
    const [lastChecked, setLastChecked] = useState<Date | null>(null);
    const [refreshKey, setRefreshKey] = useState(0);   //bump to force an immediate re-check

    useEffect(() => {
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout> | null = null;
        let controller: AbortController | null = null;

        const check = async () => {
            controller = new AbortController();
            const abortTimer = setTimeout(() => controller?.abort(), timeout);
            let healthy = false;
            let publisherStatus: HealthStatus = "retrying";
            let subscriberStatus: HealthStatus = "retrying";
            let publisherError: string | null = null;
            let subscriberError: string | null = null;
            try {
                const response: HealthStatusLog 
                    = await dataManager.get(endpoint, { skipGlobalError: true, signal: controller.signal });
                healthy = (response?.publisher?.healthy && response?.subscriber?.healthy) ?? false;
                publisherStatus = response?.publisher?.status ?? "retrying";
                subscriberStatus = response?.subscriber?.status ?? "retrying";
                publisherError = response?.publisher?.last_error ?? null;
                subscriberError = response?.subscriber?.last_error ?? null;
            } catch (error) {
                console.warn(`Health check ${endpoint} failed:`, error);
            } finally {
                clearTimeout(abortTimer);
            }

            if (cancelled) return;
            setConnected(healthy);
            setStatus({publisher: publisherStatus, subscriber: subscriberStatus});
            setErrors({publisher: publisherError, subscriber: subscriberError});   //errors of last poll only
            setLastChecked(new Date());
            //schedule next check only after this one has finished, so requests never overlap
            timer = setTimeout(check, healthy ? interval : retryInterval);
        };

        check();

        return () => {
            cancelled = true;
            if (timer) clearTimeout(timer);
            controller?.abort();
        };
    }, [endpoint, interval, retryInterval, timeout, refreshKey]);

    const refresh = () => setRefreshKey(k => k + 1);

    return {connected, status, errors, lastChecked, refresh};
}

