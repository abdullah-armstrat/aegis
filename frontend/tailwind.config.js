/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      // "Black-and-white dossier" palette (from the approved design).
      colors: {
        ink: "#161616",
        panel: "#FAFAFA",
        fill: "#F0F0F0",
        line: "#D6D6D6",
        muted: "#595959", // 7:1 on white (was #8C8C8C, 3.4:1)
        // The single deliberate accent: amber, reserved for raised flags.
        flag: "#B7791F",
        flagbg: "#FBF6EC",
        // Quiet green, reserved for "checked, clear".
        clear: "#3F7A52",
        clearbg: "#F1F6F2",
      },
      fontFamily: {
        sans: ["Archivo", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
