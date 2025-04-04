import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: NextRequest) {
  const { query } = await req.json();

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: query }],
    });

    return NextResponse.json({ response: completion.choices[0].message.content });
  } catch (error) {
    return NextResponse.json({ error: "Error from OpenAI" }, { status: 500 });
  }
}
