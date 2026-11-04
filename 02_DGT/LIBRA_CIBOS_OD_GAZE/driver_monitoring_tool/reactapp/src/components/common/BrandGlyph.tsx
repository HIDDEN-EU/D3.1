import React from 'react';

interface BrandGlyphProps {
  color?: string;
  size?: number;
  style?: React.CSSProperties;
  className?: string;
}

/**
 * Decorative node/connector motif - a clean vector redraw of the HIDDEN logo's
 * hub-and-spoke tendrils (one central node, four satellites). Purely ornamental
 * (aria-hidden) - mirrors the background accents used on hiddenproject.eu.
 */
const BrandGlyph: React.FC<BrandGlyphProps> = ({ color = '#E4007C', size = 140, style, className }) => (
  <svg
    aria-hidden="true"
    className={className}
    width={size}
    height={size}
    viewBox="0 0 1252 1252"
    style={{ pointerEvents: 'none', ...style }}
  >
    <g fill="none" stroke={color} strokeWidth="32" strokeLinecap="round">
      <line x1="503.3" y1="472.4" x2="315.0" y2="273.3" />
      <line x1="701.8" y1="468.1" x2="854.0" y2="292.0" />
      <line x1="468.7" y1="637.6" x2="277.0" y2="718.4" />
      <line x1="710.1" y1="684.2" x2="983.1" y2="954.6" />
      <circle cx="605" cy="580" r="148" />
      <circle cx="258" cy="213" r="83" />
      <circle cx="905" cy="233" r="78" />
      <circle cx="207" cy="748" r="76" />
      <circle cx="1042" cy="1013" r="83" />
    </g>
  </svg>
);

export default BrandGlyph;
