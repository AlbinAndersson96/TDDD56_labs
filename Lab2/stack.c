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
pthread_mutex_t freelistLock = PTHREAD_MUTEX_INITIALIZER;

int lockA = 1;

void addFreeNode(freelist_t* fl, Node* node) {

  // For teardown etc
  if (fl == NULL) {
    free(node);
    return;
  }

  node->val = 0;
  node->next = fl->head;

  fl->head = node;

  fl->counter++;
  fl->ops++;
}

void preAllocateNodes(freelist_t* fl, int numNodes) {
  int i;
  for(i = 0; i < numNodes; i++)
  {
    Node *node = malloc(sizeof(Node));
    addFreeNode(fl, node);
  }
}

Node* getFreeNode(freelist_t* fl)
{

  if(fl->counter != 0)
  {
    
    Node *tmp = fl->head;  
    fl->head = fl->head->next;    
    fl->counter--;

    return tmp;
  }

  return NULL;
}

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
stack_push(stack_t* stack, freelist_t* fl, int val)
{

// #if DEBUG == 1
// printStack(stack);
// #endif

  Node* new = getFreeNode(fl);  
  if (new == NULL) {
    printf("Freelist empty, re-allocating!\n");
    new = malloc(sizeof(Node)); // chunk malloc
  }

#if NON_BLOCKING == 0
  // Implement a lock_based stack
  
  pthread_mutex_lock(&mutexLock);

  new->val = val;
  new->next = stack->head;

  stack->head = new;

  stack->counter++;
  stack->ops++;
  
  
	pthread_mutex_unlock(&mutexLock);


#elif NON_BLOCKING == 1
  
  Node *head;
  do
  {
    head = stack->head;
    
    new->val = val;
    new->next = head;

  } while(cas((uintptr_t*)&stack->head, (uintptr_t)head, (uintptr_t)new) != (uintptr_t)head);

  stack->counter++;
  stack->ops++;

#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check(stack);

  return 0;
}

int /* Return the type you prefer */
stack_pop(stack_t* stack, freelist_t* freelist)
{

// #if DEBUG == 1
//  printStack(stack);
// #endif

#if NON_BLOCKING == 0
  int retVal;
  pthread_mutex_lock(&mutexLock);
  
  if (stack->counter == 0) {
    pthread_mutex_unlock(&mutexLock);
    return -1;
  }

  retVal = stack->head->val;
  Node* temp = stack->head->next;
  
  addFreeNode(freelist, stack->head);

  stack->head = temp;

  stack->counter--;
  stack->ops++;
  
  pthread_mutex_unlock(&mutexLock);

  return retVal;

#elif NON_BLOCKING == 1
  Node *head;
  Node *new;
  do
  {
    head = stack->head; 
    new = head->next;

  } while(cas((uintptr_t*)&stack->head, (uintptr_t)head, (uintptr_t)new) != (uintptr_t)head);

  addFreeNode(freelist, head);

  stack->counter--;
  stack->ops++;

#endif

  return 0;
}

int /* Return the type you prefer */
stack_push_aba(stack_t* stack, freelist_t* fl, int val)
{

// #if DEBUG == 1
// printStack(stack);
// #endif

  Node* new = getFreeNode(fl);  
  if (new == NULL) {
    printf("Freelist empty, re-allocating!\n");
    new = malloc(sizeof(Node)); // chunk malloc
  }

#if NON_BLOCKING == 1
  
  Node *head;
  do
  {
    head = stack->head;
    
    new->val = val;
    new->next = head;

  } while(cas((uintptr_t*)&stack->head, (uintptr_t)head, (uintptr_t)new) != (uintptr_t)head);

  stack->counter++;
  stack->ops++;

#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check(stack);

  return 0;
}

struct thread_test_cas_aba_args
{
  freelist_t* fl;
  stack_t* stack;
  int id;
};
typedef struct thread_test_cas_aba_args thread_test_cas_aba_args_t;

Node *abaA;

int /* Return the type you prefer */
stack_pop_aba(void* arg)
{

// #if DEBUG == 1
//  printStack(stack);
// #endif
thread_test_cas_aba_args_t *args = (thread_test_cas_aba_args_t*) arg;
freelist_t *freelist = args->fl;
stack_t *stack = args->stack;
int id = args->id;

printf("ID: %d\n", id);

#if NON_BLOCKING == 1

  // Thread B "regular pop"
  if (id == 1) {
    printf("TID: 1 => Popping A\n");
    Node *head;
    Node *new;
    do
    {
      head = stack->head;
      abaA = stack->head;
      new = head->next;
    } while(cas((uintptr_t*)&stack->head, (uintptr_t)head, (uintptr_t)new) != (uintptr_t)head);
  }

  
    
  // All threads pop
  if (id == 1) printf("TID: 1 => Popping B\n");
  if (id == 0) printf("TID: 0 => Popping A\n");
  
  Node *head;
  Node *new;
  do
  {

    head = stack->head;
    new = head->next;

    // Thread A
    if (id == 0) {
      printf("TID: 0 => Stuck in loop until lockA unlocks\n");
      while (lockA == 1);
      printf("TID: 0 => Free from loop, lockA must be 0\n");
    }
    

  } while(cas((uintptr_t*)&stack->head, (uintptr_t)head, (uintptr_t)new) != (uintptr_t)head);

  if (id == 1) addFreeNode(freelist, head);
  

  stack->counter--;
  stack->ops++;

  stack->head = abaA;
  if (id == 1) printf("TID: 1 => setting lockA = 0\n ");
  lockA = 0;

  if (id == 0) {
    // Check for B in freelist
    printf("TID: 0 => Checking for ABA\n");
    if(stack->head->next == freelist->head)
    {
      printf("TID: 0 =>ABA Detected\n");
      args->stack = NULL; // For checking purposes
      return 1;
    }
  }

#endif

  return 0;
}