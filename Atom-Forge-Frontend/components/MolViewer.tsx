"use client";

import { useEffect, useRef, useState } from "react";
import Script from "next/script";

interface MolViewerProps {
  sdfString: string;
  width?: number;
  height?: number;
}

// Element label colors — CPK scheme adapted to the sci-fi aesthetic
const ELEM_COLORS: Record<string, string> = {
  C:  "#cccccc",   // grey
  N:  "#4fc3f7",   // cyan-blue
  O:  "#ef5350",   // red
  S:  "#ffd54f",   // yellow
  F:  "#69f0ae",   // green
  Cl: "#76ff03",   // lime
  Br: "#ff7043",   // orange
  I:  "#ce93d8",   // purple
  P:  "#ffb74d",   // amber
};

export default function MolViewer({
  sdfString,
  width  = 244,
  height = 244,
}: MolViewerProps) {
  const viewerRef   = useRef<HTMLDivElement>(null);
  const [scriptLoaded, setScriptLoaded] = useState(false);

  useEffect(() => {
    if ((window as any).$3Dmol) {
      setScriptLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!scriptLoaded || !sdfString || !viewerRef.current) return;

    const timer = setTimeout(() => {
      try {
        const win = window as any;
        if (!win.$3Dmol) return;

        // Clear previous render
        viewerRef.current!.innerHTML = "";

        const viewer = win.$3Dmol.createViewer(viewerRef.current, {
          backgroundColor: "#020c14",
        });

        // Load molecule
        viewer.addModel(sdfString, "sdf");

        // Stick + sphere style
        viewer.setStyle({}, {
          stick: {
            radius:      0.14,
            colorscheme: "default",
          },
          sphere: {
            scale:       0.22,
            colorscheme: "default",
          },
        });

        // ── Atom labels ──────────────────────────────────
        const atoms = viewer.getModel(0).selectedAtoms({});

        atoms.forEach((atom: any) => {
          // Skip hydrogens — too many, clutters the view
          if (atom.elem === "H") return;

          const color = ELEM_COLORS[atom.elem] ?? "#ffffff";

          viewer.addLabel(atom.elem, {
            position: {
              x: atom.x,
              y: atom.y,
              z: atom.z,
            },
            fontColor:         color,
            fontSize:          11,
            fontOpacity:       1,
            bold:              true,
            borderThickness:   0,
            backgroundColor:   "#020c14",
            backgroundOpacity: 0.55,
            inFront:           true,
            alignment:         "center",
          });
        });
        // ─────────────────────────────────────────────────

        viewer.zoomTo();
        viewer.spin("y", 0.4);
        viewer.render();

      } catch (err) {
        console.error("3Dmol render error:", err);
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [scriptLoaded, sdfString]);

  return (
    <>
      <Script
        src="https://3dmol.org/build/3Dmol-min.js"
        strategy="afterInteractive"
        onLoad={() => setScriptLoaded(true)}
      />
      <div
        ref={viewerRef}
        style={{
          width:        `${width}px`,
          height:       `${height}px`,
          position:     "relative",
          borderRadius: "14px",
          overflow:     "hidden",
          background:   "#020c14",
          border:       "1px solid rgba(0,229,255,0.15)",
          boxShadow:    "0 0 40px rgba(0,229,255,0.06) inset",
        }}
      />
    </>
  );
}