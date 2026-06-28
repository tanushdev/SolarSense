import { cn } from "@/lib/utils"

export function Card({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("bg-bug-card border border-bug-hairline", className)} {...props}>{children}</div>
}

export function CardHeader({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 pt-5 pb-3", className)} {...props}>{children}</div>
}

export function CardTitle({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("display-sm text-bug-text", className)} {...props}>{children}</h3>
}

export function CardContent({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 pb-5", className)} {...props}>{children}</div>
}
