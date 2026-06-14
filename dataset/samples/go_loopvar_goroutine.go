package worker

import (
	"fmt"
	"sync"
)

func ProcessAll(items []string) {
	var wg sync.WaitGroup
	for _, item := range items {
		wg.Add(1)
		go func() {
			defer wg.Done()
			fmt.Println("processing", item)
		}()
	}
	wg.Wait()
}
