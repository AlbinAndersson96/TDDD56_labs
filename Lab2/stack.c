/**
 * Ask about calling conv for CAS, what are we missing?
 * Freelist?
 * Tests and multi-threading. Lab instructions and code/comments don't add up?
 */







/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

pthread_mutex_t mutexLock = PTHREAD_MUTEX_INITIALIZER;

void printStack(stack_t* stack) 
{
  pthread_mutex_lock(&mutexLock);
  if (stack->head == NULL) {
    printf("List is empty!\n");
    pthread_mutex_unlock(&mutexLock);
    return;
  }

  Node* current = stack->head;
  printf("head->%d", current->val);
  while(current->next != NULL) { 
    current = current->next;
    printf("->%d", current->val);
  }
  printf("\n");

  pthread_mutex_unlock(&mutexLock);
}


int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(stack_t* stack, int val)
{

#if DEBUG == 1
printStack(stack);
#endif

#if NON_BLOCKING == 0
  // Implement a lock_based stack

  Node* new = malloc(sizeof(Node));

  pthread_mutex_lock(&mutexLock);

  new->val = val;
  new->next = stack->head;

  stack->head = new;

  stack->counter++;
  stack->ops++;
  
	pthread_mutex_unlock(&mutexLock);


#elif NON_BLOCKING == 1
  
  Node *new = malloc(sizeof(Node));
  Node *head = stack->head;
  new->val = val;
  new->next = head;
    
  printf("%lud\n",cas(&stack->head->val, head->val, new->val));  // Ask about calling conv for CAS

#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check(stack);

  return 0;
}

int /* Return the type you prefer */
stack_pop(stack_t* stack)
{

#if DEBUG == 1
 printStack(stack);
#endif

#if NON_BLOCKING == 0
  int retVal;
  pthread_mutex_lock(&mutexLock);
  
  if (stack->counter == 0) {
    pthread_mutex_unlock(&mutexLock);
    return -1;
  }

  retVal = stack->head->val;
  Node* temp = stack->head->next;
  free(stack->head);
  stack->head = temp;

  stack->counter--;
  stack->ops++;
  
  pthread_mutex_unlock(&mutexLock);

  return retVal;

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
#endif

  return 0;
}

