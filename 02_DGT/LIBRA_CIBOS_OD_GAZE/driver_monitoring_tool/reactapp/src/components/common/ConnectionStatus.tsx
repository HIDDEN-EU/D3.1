import { CheckCircleFilled, CloseCircleFilled, ExclamationCircleFilled, ExclamationCircleOutlined, InfoCircleOutlined, SyncOutlined, } from "@ant-design/icons";
import type { BadgeProps } from "antd";
import { Badge, Popover, Typography } from "antd";
import { useJanusSession } from "../../context/JanusContext";
import { useRos } from "../../context/RosContext";
import { useHealthCheck } from "../../hooks/useHealthCheck";
import { STATUS_MESSAGE, type HealthStatus } from "../../types/healthStatusTypes";

const { Text } = Typography;

const BADGE_STATUS: Record<HealthStatus, BadgeProps["status"]> = {
    connected: "success",
    subscribed: "success",
    connecting: "processing",
    retrying: "warning",
    error: "error",
    invalid_data: "error",
    disconnected: "default",
};

interface StatusInfoProps {
    label: string;
    badge: BadgeProps["status"];
    text: string;
    sub?: boolean;  //indented detail row
    error?: string | null;  //last_error of sub row, shown next to badge
}

const StatusIcon: React.FC<{ status: BadgeProps["status"] }> = ({ status }) => {
    switch (status) {
        case "success":
            return <CheckCircleFilled style={{ color: "#52c41a", fontSize: 14 }} />;
        case "processing":
            return <SyncOutlined spin style={{ color: "#faad14", fontSize: 14 }} />;
        case "warning":
            return <ExclamationCircleFilled style={{ color: "#faad14", fontSize: 14 }} />;
        default:
            return <CloseCircleFilled style={{ color: "#ff4d4f", fontSize: 14 }} />;
    }
};

//main rows use icons, sub rows use badge dots
const StatusInfo: React.FC<StatusInfoProps> = ({ label, badge, text, sub, error }) => (
    <>
        {sub
            ? <span style={{ paddingLeft: 20 }}>    {/* icon width + gap --> aligns with main labels */}
                <Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>
              </span>
            : <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 13, alignSelf: "center" }}>
                <StatusIcon status={badge} />
                {label}
              </span>}
        {sub
            ? <span style={{ display: "inline-flex", alignItems: "baseline", gap: 8, minWidth: 0 }}>
                <Badge status={badge} text={<span style={{ fontSize: 12, whiteSpace: "nowrap" }}>{text}</span>} />
                {error &&
                    <Text type="danger" style={{ fontSize: 11, maxWidth: 220, whiteSpace: "normal", overflowWrap: "anywhere" }}>
                        {error}
                    </Text>}
              </span>
            : <span style={{ fontSize: 13, alignSelf: "center" }}>{text}</span>}  {/* single line --> center with icon */}
    </>
);

const ConnectionStatus: React.FC = () => {
    const janusSession = useJanusSession();
    const ros = useRos();
    const janusConnected = !!janusSession?.janus;
    const rosConnected = !!ros?.connected;
    const fastApi = useHealthCheck("/health/rosbridge");
    const fastApiConnected = fastApi.connected;
    const hasConnectionError = !janusConnected || !rosConnected || !fastApiConnected;

    const pubError = BADGE_STATUS[fastApi.status.publisher] === "error";
    const subError = BADGE_STATUS[fastApi.status.subscriber] === "error";
    const fastApiSummary = fastApi.connected
        ? { badge: "success" as const, text: "Connected" }
        : pubError || subError
            ? { badge: "error" as const, text: "" }
            : { badge: "processing" as const, text: "" };

    const content = (
        <div style={{ minWidth: 260 }}>
            <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", columnGap: 24, rowGap: 6, alignItems: "baseline" }}>
                <StatusInfo
                    label="Janus"
                    badge={janusConnected ? "success" : janusSession?.state ? "processing" : "error"}
                    text={janusConnected ? "Connected" : janusSession?.state || "Disconnected"}
                />
                <StatusInfo label="ROS" badge={rosConnected ? "success" : "error"} text={rosConnected ? "Connected" : "Disconnected"} />
                <StatusInfo label="FastAPI (rosbridge)" badge={fastApiSummary.badge} text={fastApiSummary.text} />
                <StatusInfo sub label="Publisher" badge={BADGE_STATUS[fastApi.status.publisher]} text={STATUS_MESSAGE[fastApi.status.publisher]} />
                <StatusInfo sub label="Subscriber" badge={BADGE_STATUS[fastApi.status.subscriber]} text={STATUS_MESSAGE[fastApi.status.subscriber]} />
            </div>
        </div>
    );

    return (
        <Popover /*title="Connections"*/ content={content} trigger={["hover", "click"]} placement="bottomRight">
            {hasConnectionError
                ? <ExclamationCircleOutlined style={{ fontSize: "25px", color: "#cf761c", cursor: "pointer" }} />
                : <InfoCircleOutlined style={{ fontSize: "25px", color: "#1a36a5", cursor: "pointer" }} />}
        </Popover>
    )
};

export default ConnectionStatus;