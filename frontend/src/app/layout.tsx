import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StockPulse — AI Stock Analysis",
  description: "AI-powered stock analysis and signal generation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
