// import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';
export const maxDuration = 30;

export async function POST(req: Request) {
    console.error("worked")
    const { messages } = await req.json();

    const result = streamText({
        model: openai('gpt-4-turbo'),
        system: 'You are a helpful assistant.',
        messages,
    });

    return result.toDataStreamResponse();
}
