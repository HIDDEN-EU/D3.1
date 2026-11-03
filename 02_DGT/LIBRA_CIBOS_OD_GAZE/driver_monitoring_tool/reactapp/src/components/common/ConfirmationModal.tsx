import { CloseOutlined, DoubleRightOutlined, RedoOutlined, SaveOutlined } from "@ant-design/icons";
import { Button, Modal, Space, Typography } from "antd";
import { useEffect } from "react";
import { ShortcutKey } from "./ShortcutKey";

const { Text } = Typography;

interface ModalProps {
    open: boolean;
    onRetake?: () => void;
    onMove?: () => void;
    onSave?: () => Promise<void>;
    onCancel?: () => void;
    getContainer?: HTMLElement | (() => HTMLElement) | false;
}

const ConfirmationModal: React.FC<ModalProps> = ({
    open,
    onRetake,
    onMove,
    onSave,
    onCancel,
    getContainer
}) => {

    useEffect(() => {
        if (!open) return;
        
        const handlers: Record<string, (() => void) | undefined> = {
            r: onRetake,
            n: onMove,
            c: onCancel,
            s: onSave
        };

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
            const handler = handlers[e.key.toLowerCase()];
            if (!handler) return;

            e.preventDefault();
            handler();
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [open, onRetake, onMove, onSave, onCancel]);

    return (
        <Modal 
            open={open} 
            centered 
            width="fit-content"
            getContainer={getContainer} 
            closable={false}
            style={{ cursor: "default" }}
            footer={null}
            destroyOnHidden
        >
            <Space orientation="vertical" size="middle" wrap>
                <Text>
                    Choose an action:
                </Text>
                <Space size="middle" wrap>
                    {onRetake && <Button icon={<RedoOutlined />} onClick={onRetake}>
                        Retake point <ShortcutKey keyName="R" />
                    </Button>}
                    {onMove && <Button color="default" variant="solid" icon={<DoubleRightOutlined />} onClick={onMove}>
                        Next point <ShortcutKey keyName="N" />
                    </Button>}
                    {onCancel && <Button icon={<CloseOutlined />} onClick={onCancel}>
                        Cancel process <ShortcutKey keyName="C" />
                    </Button>}
                    {onSave && <Button color="geekblue" variant="solid" icon={<SaveOutlined />} onClick={onSave}>
                        Save process <ShortcutKey keyName="S" />
                    </Button>}
                </Space>
            </Space>
        </Modal>
    )
};

export default ConfirmationModal;