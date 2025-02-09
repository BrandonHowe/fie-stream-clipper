import { dlopen } from 'bun:ffi';
import path from 'path';

// Load the DLL
const videoReviewDll = dlopen(path.resolve('./foreign/build/Release/video_review.dll'), {
  add: { returns: 'i32', args: ['i32', 'i32'] },
});

// Call the function
const result = videoReviewDll.symbols.add(5, 7);
console.log('Result of add(5, 7):', result);