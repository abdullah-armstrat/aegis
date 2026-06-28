// The three-state signal as a 42×42 ICON BOX (left-rail), exactly per the dossier design.
// Distinguished by SHAPE + the box treatment before colour:
//   fired        → pale amber box (#FBF1DD), amber flag icon (#B7791F)
//   not_assessed → white box with a DASHED border (#AEB6C0), slate broken-ring icon (#5B6470)
//   clear        → pale green box (#E6EFE8), green ring+tick (#3F7A52)
// `flag.status` from the backend: "fired" | "clear" | "not_assessed".

export const STATUS_META = {
  fired: { label: "Flag raised", labelColor: "#8A5A12" },
  not_assessed: { label: "Couldn’t check", labelColor: "#4D5662" },
  clear: { label: "Checked, clear", labelColor: "#3F7A52" },
};

export default function Signal({ status }) {
  if (status === "fired") {
    return (
      <div
        className="flex h-[42px] w-[42px] items-center justify-center"
        style={{ background: "#FBF1DD" }}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <rect x="5" y="3" width="2" height="18" rx="1" fill="#B7791F" />
          <path d="M7 4.2 L19 8 L7 11.8 Z" fill="#B7791F" />
        </svg>
      </div>
    );
  }

  if (status === "clear") {
    return (
      <div
        className="flex h-[42px] w-[42px] items-center justify-center"
        style={{ background: "#E6EFE8" }}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="8.5" stroke="#3F7A52" strokeWidth="2" />
          <path
            d="M8 12.2 L11 15.2 L16.2 9"
            stroke="#3F7A52"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  }

  // not_assessed: white box, dashed border, broken ring — deliberately incomplete.
  return (
    <div
      className="flex h-[42px] w-[42px] items-center justify-center bg-white"
      style={{ border: "1px dashed #AEB6C0" }}
    >
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle
          cx="12"
          cy="12"
          r="8.5"
          stroke="#5B6470"
          strokeWidth="2"
          strokeDasharray="3.2 3.2"
        />
        <line x1="8" y1="12" x2="16" y2="12" stroke="#5B6470" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </div>
  );
}
