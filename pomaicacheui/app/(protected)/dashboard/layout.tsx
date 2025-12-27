export const metadata = {
    title: 'Dashboard - Pomaicache',
    description: 'Access your personalized dashboard to manage your Pomaicache settings and data.',
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
    return <section>{children}</section>
}