import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(iso: string) {
  if (!iso) return "—"
  const d = new Date(iso)
  return d.toLocaleString("en-IN", { timeZone: "UTC", hour12: false })
}

export function formatTime(iso: string) {
  if (!iso) return "—"
  const d = new Date(iso)
  return d.toLocaleTimeString("en-IN", { timeZone: "UTC", hour12: false })
}

export function classColor(cls: string) {
  const map: Record<string, string> = { X: "text-bug-warning", M: "text-bug-body-strong", C: "text-bug-muted", B: "text-bug-muted-soft", A: "text-bug-muted-soft" }
  return map[cls?.toUpperCase()] ?? "text-bug-text"
}
