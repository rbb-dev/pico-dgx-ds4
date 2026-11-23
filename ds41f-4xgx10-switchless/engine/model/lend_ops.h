// The pool's side of lend::Lender (lend.h): the numbers, who holds them, the reservations, the cut. Shared by window.cu and
// tenant.cu.
#pragma once
#include "model/impl.h"

namespace model {

struct Model::Impl::LendOps {
  Model::Impl& m;
  size_t numbers() const { return m.calloc->numbers(); }
  bool vacate(size_t top) { return m.core->vacate((uint32_t)top) != 0; }
  bool make_room(size_t removed) { return m.core->make_room(removed); }
  // The handle count the cut leaves is checked by the caller's proof (the pool's own count before and after, against the
  // arithmetic): a rank that disagrees is named there, and every rank answers alike.
  bool release(size_t top, int) {
    m.pool_cut(top);
    return true;
  }
  bool grow_all(size_t more) { return m.pool_grow_all(more); }
};

}  // namespace model
