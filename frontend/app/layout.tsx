import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedBot",
  description: "Ask questions and get answers grounded in the medical encyclopedia.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
