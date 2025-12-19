import "./globals.css";
import Providers from "./provider";
import "remixicon/fonts/remixicon.css";

export const metadata = {
  title: "Pomai Cache UI",
  description: "Admin dashboard"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}