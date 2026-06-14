export class Uploader {
  private queue: string[] = [];

  enqueue(file: string): void {
    this.queue.push(file);
  }

  process(handler: (f: string) => Promise<void>): void {
    this.queue.forEach((file) => {
      handler(file);
    });
    this.queue = [];
  }
}
