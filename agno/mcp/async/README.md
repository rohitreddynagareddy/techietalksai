# Python Asynchronous Programming Examples

This directory contains Python scripts demonstrating various aspects of asynchronous programming in Python.

## Basic Concepts

- **async-1.py**: Basic async function implementation with `asyncio.sleep()` demonstrating non-blocking delay
- **async-1a.py**: Running multiple coroutines concurrently with `asyncio.gather()`
- **async-1b.py**: Threading implementation for comparison with async methods
- **async-1c.py**: Sequential blocking implementation for comparison
- **async-1d.py**: Direct comparison of synchronous vs asynchronous performance with timing
- **async-1e.py**: Demonstrates background tasks with `asyncio.create_task()`

## Advanced Async Patterns

- **async-2.py**: Shows how tasks run concurrently regardless of their order in `gather()`
- **async-3-with.py**: Implements async context manager with `__aenter__` and `__aexit__`
- **async-3-withb.py**: Regular synchronous context manager for comparison
- **async-4-iter.py**: Creates an async iterable object using `__aiter__` and `__anext__`

## Real-world Application

- **sse-s.py**: Server-side events implementation with FastAPI
- **sse-c.py**: Client that consumes SSE events asynchronously

## Key Async Concepts Demonstrated

- Coroutines defined with `async def`
- Awaiting with `await`
- Task scheduling with `asyncio.gather()` and `asyncio.create_task()`
- Event loop management with `asyncio.run()`
- Async context management with `async with`
- Async iteration with `async for`
- Comparing threading with async for I/O-bound operations