import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "Pulso TransMi · MLOps", description: "Monitoreo del modelo y data drift" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="es"><body>{children}</body></html>;
}
