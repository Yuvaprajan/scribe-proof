import type { Metadata } from "next";
import { DM_Sans, Instrument_Serif, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const display = Instrument_Serif({
  subsets: ["latin"],
  variable: "--font-display",
  weight: "400",
});

const sans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["400", "500", "600", "700"],
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "ScribeProof — Agentic handwriting extraction with visual grounding",
  description:
    "Parse messy handwriting into grounded, auditable text. Every word has a crop, OCR hypotheses, confidence, and an explicit decision.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${display.variable} ${sans.variable} ${mono.variable} font-sans antialiased text-slateink-900`}
      >
        <div className="min-h-screen">
          <header className="sticky top-0 z-40 border-b border-slateink-200/80 bg-white/85 backdrop-blur-md">
            <div className="mx-auto flex h-14 max-w-[1400px] items-center justify-between px-4 sm:px-6">
              <div className="flex items-center gap-8">
                <a href="/" className="flex items-center gap-2.5">
                  <span className="flex h-7 w-7 items-center justify-center rounded-md bg-brand-700 text-xs font-bold text-white">
                    SP
                  </span>
                  <span className="text-[15px] font-semibold tracking-tight text-slateink-900">
                    ScribeProof
                  </span>
                </a>
                <nav className="hidden items-center gap-1 text-sm text-slateink-600 md:flex">
                  <a href="/#product" className="toolbar-btn">
                    Product
                  </a>
                  <a href="/playground" className="toolbar-btn">
                    Playground
                  </a>
                  <a href="/#grounding" className="toolbar-btn">
                    Grounding
                  </a>
                </nav>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href="/playground"
                  className="rounded-lg bg-brand-700 px-3.5 py-1.5 text-sm font-semibold text-white shadow-soft transition hover:bg-brand-800"
                >
                  Open Playground
                </a>
              </div>
            </div>
          </header>
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
