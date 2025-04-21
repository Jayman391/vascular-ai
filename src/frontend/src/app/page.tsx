'use client';

import { useChat } from '@ai-sdk/react';

export default function Page() {
    const { messages, input, handleInputChange, handleSubmit } = useChat({});

    return (
        <main className="flex flex-col gap-14 items-center justify-center w-full h-full">
            <h1 className="text-6xl font-bold">
                PMLLM
            </h1>
            {messages.map(message => (
                <div key={message.id}>
                    {message.role === 'user' ? 'User: ' : 'AI: '}
                    {message.content}
                </div>
            ))}

            <form className="flex flex-row gap-4" onSubmit={handleSubmit}>
                <input className="border-b-6 w-128 outline-none" name="prompt" value={input} onChange={handleInputChange} />
                <button className="font-bold bg-blue-700 text-white rounded-md shadow-sm px-4 py-2" type="submit">Submit</button>
            </form>
        </main>
    );
}
