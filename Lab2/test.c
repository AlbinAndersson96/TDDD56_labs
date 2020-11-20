/*
 * test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "test.h"
#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'...\n ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

/* Helper function for measurement */
double timediff(struct timespec *begin, struct timespec *end)
{
	double sec = 0.0, nsec = 0.0;
   if ((end->tv_nsec - begin->tv_nsec) < 0)
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec  - 1);
      nsec = (double)(end->tv_nsec - begin->tv_nsec + 1000000000);
   } else
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec );
      nsec = (double)(end->tv_nsec - begin->tv_nsec);
   }
   return sec + nsec / 1E9;
}

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

#ifndef NDEBUG
int
assert_fun(int expr, const char *str, const char *file, const char* function, size_t line)
{
	if(!(expr))
	{
		fprintf(stderr, "[%s:%s:%zu][ERROR] Assertion failure: %s\n", file, function, line, str);
		abort();
		// If some hack disables abort above
		return 0;
	}
	else
		return 1;
}
#endif


stack_t *stack;
data_t data;

#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;

  // Per thread freelist
  freelist_t *fl = malloc(sizeof(freelist_t));
  preAllocateNodes(fl, MAX_PUSH_POP / NB_THREADS);

    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    { 
      stack_push(stack, fl, i);
      
    }

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
      {
        stack_pop(stack, fl);
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

    return NULL;
  }
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;

  // Per thread freelist
  freelist_t *fl = malloc(sizeof(freelist_t));
  preAllocateNodes(fl, MAX_PUSH_POP / NB_THREADS);

  
  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    { 
      //d -> %d\n", args->id, i);
      stack_push(stack, fl, i);
      
    }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#endif

// Declares
void* thread_test_lock() {
  freelist_t *fl = malloc(sizeof(freelist_t));
  preAllocateNodes(fl, MAX_PUSH_POP);

  for (int i = 0; i < MAX_PUSH_POP; ++i)
    {
      stack_push(stack, fl, i);
    }
    
  for (int i = 0; i < MAX_PUSH_POP/2; ++i)
    {
      stack_pop(stack, fl);
    }

  return NULL;
}

int test_lock() {
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  int success;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (int i = 0; i < NB_THREADS; ++i)
  { 
    pthread_create(&thread[i], &attr, &thread_test_lock, NULL);
  }

  for (int i = 0; i < NB_THREADS; ++i)
  {
    pthread_join(thread[i], NULL);
  }

  success = assert(stack->counter == (size_t)(NB_THREADS * MAX_PUSH_POP / 2));

  if (!success)
  {
    printf("Got %i, expected %i. \n", stack->counter, NB_THREADS * MAX_PUSH_POP / 2);
    printf("Is MAX_PUSH_POP an even number?\n");
  }

  if (stack->head == NULL) {
    printf("List is empty!\n");
    return success;
  }

  Node* current = stack->head;
  printf("head->%d", current->val);
  while(current->next != NULL) { 
    current = current->next;
    printf("->%d", current->val);
  }
  printf("\n");

  return success;
}


/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{

}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  Node* head = malloc(sizeof(Node));
  //head->val = 0;
  head->val = -1;
  head->next = NULL;

  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_t));


  // Reset explicitely all members to a well-known initial value
  // For instance (to be deleted as your stack design progresses):
  stack->ops = 0;
  stack->counter = 0;
  stack->head = head;
  
}

void
test_teardown()
{
  printf("Started teardown\n");

  // Do not forget to free your stacks after each test
  while(stack->counter != 0) 
  { 
    stack_pop(stack, NULL);
  }

  // to avoid memory leaks
  free(stack->head);
  free(stack);

   //while(freelist->counter != 0) 
   //{ 
   //  Node* t = getFreeNode(freelist);
     
   //  free(t);
   //}

  // free(freelist);
}

void
test_finalize()
{
  // Destroy properly your test batch
}

int
test_push_safe()
{

  // Per thread freelist
  freelist_t *fl = malloc(sizeof(freelist_t));
  preAllocateNodes(fl, MAX_PUSH_POP);
  
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it

  // Do some work
  stack_push(stack, fl, 0);
  stack_push(stack, fl, 1);
  stack_push(stack, fl, 2);
  stack_push(stack, fl, 3);


  // check if the stack is in a consistent state
  int res = assert(stack_check(stack));

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // Now, the test succeeds
  return res && assert(stack->counter == 4 && stack->ops == 4); //LOOK AT THIS LATER
}

int
test_pop_safe()
{
  freelist_t *fl = malloc(sizeof(freelist_t));
  preAllocateNodes(fl, MAX_PUSH_POP);

  // Same as the test above for parallel pop operation
  stack_push(stack, fl, 0);
  stack_push(stack, fl, 1);
  stack_push(stack, fl, 2);
  stack_push(stack, fl, 3);

  stack_pop(stack, fl);
  stack_pop(stack, fl);
  stack_pop(stack, fl);
  stack_pop(stack, fl);
  
  // For now, this test always fails
  return assert(stack->counter == 0 && stack->ops == 8);
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

int
test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_t thread[ABA_NB_THREADS];
  int success, aba_detected = 0;
  // Write here a test for the ABA problem
  success = aba_detected;
  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = assert(counter == (size_t)(NB_THREADS * MAX_PUSH_POP));

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_lock);
  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  //test_run(test_aba);

  //test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];
  pthread_attr_init(&attr);

  Node* head = malloc(sizeof(Node));
  head->val = -1;
  head->next = NULL;

  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_t));

#if MEASURE == 1
  // test_setup();

  // for(int i = 0; i < MAX_PUSH_POP; i++) {
  //   stack_push(stack, i);
  // }
  
#elif MEASURE == 2
  //test_setup();

#endif

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
        printf("Thread %d time: %f\n", i, timediff(&t_start[i], &t_stop[i]));
    }
#endif

  return 0;
}
