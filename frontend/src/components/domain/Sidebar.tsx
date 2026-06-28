import { NavLink } from "react-router-dom"
import { cn } from "@/lib/utils"

const links = [
  { to: "/", label: "MISSION CONTROL" },
  { to: "/live-monitor", label: "LIVE MONITOR" },
  { to: "/forecasts", label: "FORECASTS" },
  { to: "/historical-events", label: "HISTORY" },
  { to: "/model-performance", label: "TELEMETRY" },
  { to: "/datasets", label: "DATA ARCHIVE" },
  { to: "/validation", label: "VERIFICATION" },
]

export function TopNav() {
  return (
    <nav className="flex items-stretch border-b border-bug-hairline bg-bug-bg px-6">
      {links.map(({ to, label }) => (
        <NavLink key={to} to={to} end={to === "/"}
          className={({ isActive }) => cn(
            "flex-1 flex items-center justify-center py-3 mono-label transition-colors border-b-2 -mb-[1px] text-center",
            isActive ? "text-bug-text border-bug-text" : "text-bug-muted border-transparent hover:text-bug-body"
          )}>
          {label}
        </NavLink>
      ))}
    </nav>
  )
}
