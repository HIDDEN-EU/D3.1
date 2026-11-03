export const ShortcutKey: React.FC<{ keyName: string }> = ({ keyName }) => (
    <kbd
        style={{
            marginInlineStart: 6,
            padding: "0 5px",
            fontSize: "0.85em",
            color: "inherit",
            background: "color-mix(in srgb, currentColor 8%, transparent)",
            border: "1px solid color-mix(in srgb, currentColor 40%, transparent)",
            borderBottomWidth: 2,
            borderRadius: 3,
        }}
    >
        {keyName}
    </kbd>
);