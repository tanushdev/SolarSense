import { Outlet } from 'react-router-dom'
import { TopNav } from '@/components/domain/Sidebar'

export function AppLayout() {
  return (
    <div className="h-screen flex flex-col bg-bug-bg text-bug-body">
      <header className="h-14 flex items-center justify-center border-b border-bug-hairline bg-bug-bg shrink-0">
        <span className="wordmark text-bug-text">SOLARSENSE</span>
      </header>
      <TopNav />
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        <Outlet />
      </main>
    </div>
  )
}
