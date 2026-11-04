import { Button } from "antd";
import { CloseOutlined } from "@ant-design/icons";
import { ShortcutKey } from "../common/ShortcutKey";

interface FullscreenHoverBannerProps {
    visible: boolean;
    onCancel: () => void;
    label?: string;
}

const FullscreenHoverBanner: React.FC<FullscreenHoverBannerProps> = ({
    visible, 
    onCancel, 
    label = "Stop Process"
}) => (
    <div
        style={{
            position: "absolute",
            top: 0, left: 0, right: 0,
            display: "flex",
            justifyContent: "flex-end",
            padding: "16px 24px",
            zIndex: 1000,
            background: "linear-gradient(to bottom, rgba(0,0,0,0.55), rgba(0,0,0,0))",
            opacity: visible ? 1 : 0,
            transform: visible ? "translateY(0)" : "translateY(-12px)",
            transition: "opacity 0.25s ease, transform 0.25s ease",
            pointerEvents: visible ? "auto" : "none",
        }}
    >
        <Button color="danger" variant="solid" icon={<CloseOutlined />} onClick={onCancel}>
            {label} <ShortcutKey keyName="esc"/>
        </Button>
    </div>
);

export default FullscreenHoverBanner;
