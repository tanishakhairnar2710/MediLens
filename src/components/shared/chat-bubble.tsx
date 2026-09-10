"use client";

import { motion } from "framer-motion";
import { Activity } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import type { ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex items-end gap-2.5",
        isUser && "flex-row-reverse"
      )}
    >
      {isUser ? (
        <Avatar className="h-7 w-7 shrink-0">
          <AvatarFallback className="text-[11px]">
            AM
          </AvatarFallback>
        </Avatar>
      ) : (
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-white">
          <Activity className="h-3.5 w-3.5" />
        </span>
      )}

      <div className="max-w-[78%] sm:max-w-[65%]">
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "rounded-br-md bg-primary text-white"
              : "rounded-bl-md border border-border bg-white text-foreground"
          )}
        >
          {isUser ? (
            <div className="whitespace-pre-wrap">
              {message.content}
            </div>
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ children }) => (
                    <h1 className="mb-3 mt-1 text-base font-semibold">
                      {children}
                    </h1>
                  ),

                  h2: ({ children }) => (
                    <h2 className="mb-2 mt-3 text-base font-semibold first:mt-0">
                      {children}
                    </h2>
                  ),

                  h3: ({ children }) => (
                    <h3 className="mb-1.5 mt-3 text-sm font-semibold first:mt-0">
                      {children}
                    </h3>
                  ),

                  p: ({ children }) => (
                    <p className="mb-2 last:mb-0">
                      {children}
                    </p>
                  ),

                  ul: ({ children }) => (
                    <ul className="mb-2 list-disc space-y-1 pl-5">
                      {children}
                    </ul>
                  ),

                  ol: ({ children }) => (
                    <ol className="mb-2 list-decimal space-y-1 pl-5">
                      {children}
                    </ol>
                  ),

                  li: ({ children }) => (
                    <li className="pl-1">
                      {children}
                    </li>
                  ),

                  strong: ({ children }) => (
                    <strong className="font-semibold">
                      {children}
                    </strong>
                  ),

                  em: ({ children }) => (
                    <em>{children}</em>
                  ),

                  blockquote: ({ children }) => (
                    <blockquote className="my-2 border-l-2 border-border pl-3 italic">
                      {children}
                    </blockquote>
                  ),

                  table: ({ children }) => (
                    <div className="my-3 overflow-x-auto rounded-md border border-border">
                      <table className="w-full border-collapse text-xs">
                        {children}
                      </table>
                    </div>
                  ),

                  thead: ({ children }) => (
                    <thead className="bg-muted/50">
                      {children}
                    </thead>
                  ),

                  th: ({ children }) => (
                    <th className="border-b border-border px-3 py-2 text-left font-semibold">
                      {children}
                    </th>
                  ),

                  td: ({ children }) => (
                    <td className="border-b border-border px-3 py-2 align-top last:border-b-0">
                      {children}
                    </td>
                  ),

                  hr: () => (
                    <hr className="my-3 border-border" />
                  ),

                  code: ({ children }) => (
                    <code className="rounded bg-muted px-1 py-0.5 text-xs">
                      {children}
                    </code>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        <span
          className={cn(
            "mt-1 block text-[11px] text-muted",
            isUser && "text-right"
          )}
        >
          {message.timestamp}
        </span>
      </div>
    </motion.div>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex items-end gap-2.5">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-white">
        <Activity className="h-3.5 w-3.5" />
      </span>

      <div className="flex items-center gap-1 rounded-2xl rounded-bl-md border border-border bg-white px-4 py-3.5">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-muted"
            animate={{ y: [0, -4, 0] }}
            transition={{
              duration: 0.9,
              repeat: Infinity,
              delay: i * 0.15,
            }}
          />
        ))}
      </div>
    </div>
  );
}
