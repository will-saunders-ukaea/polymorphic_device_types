#include <iostream>
#include <CL/sycl.hpp>
#include <memory>
#include <numeric>

using namespace cl;

// ------------ REACTION BASE CLASS -----------------

/**
 * Base reaction type that all reactions get cast to. Defines the interface for
 * reactions.
 */
struct Reaction {
  virtual inline void react(sycl::queue &, std::vector<double> &) {}
};

// ------------ CRTP BASE CLASS -----------------

/**
 * This is a templated type that inherits from Reaction such that
 * specialisations can define their reaction data but the instance can be cast
 * to the base type such that react can be called with dynamic polymorhism.
 */
template <typename DERIVED_REACTION>
struct ReactionBase : Reaction {
  inline void react(sycl::queue & queue, std::vector<double> & io) override {
    const auto& underlying = static_cast<DERIVED_REACTION&>(*this);
    auto device_type = underlying.get_device_data();
    const int N = io.size();
    sycl::buffer<double, 1> b_io(io.data(), io.size());

    queue.submit([&] (sycl::handler& h) {
      auto a_io = b_io.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(
        sycl::range<1>(N),
        [=] (auto id) {
          device_type.apply(a_io[id]);
        });
    }).wait_and_throw();   


  }
};

// ------------ REACTION A -----------------

struct DeviceReactionA {
  double a;
  inline void apply(double & d) const {
    d *= a;
  }
};

struct ReactionA : ReactionBase<ReactionA> {
  DeviceReactionA data;
  DeviceReactionA get_device_data() const{
    return this->data;
  }
  ReactionA(const double a){
    // create data in the device type for this reaction
    this->data.a = a;
  }
};


// ------------ REACTION B -----------------

struct DeviceReactionB {
 int b;
  inline void apply(double & d) const{
    d += b;
  }
};

struct ReactionB : ReactionBase<ReactionB> {
  DeviceReactionB data;
  DeviceReactionB get_device_data() const{
    return this->data;
  }
  ReactionB(const int b){
    // create data in the device type for this reaction
    this->data.b = b;
  }
};

// ------------ HELPER FUNCTION ------------
//
template<typename REACTION, typename... ARGS>
inline std::shared_ptr<Reaction> make_reaction(ARGS... args){
  auto r = std::make_shared<REACTION>(args...);
  return std::dynamic_pointer_cast<Reaction>(r);
}


int main(int argc, char **argv) {

  sycl::device device{};
  std::cout << "Using " << device.get_info<sycl::info::device::name>() << std::endl;
  sycl::queue queue{device};

  const int N = 32;
  std::vector<double> d(N);
  std::iota(d.begin(), d.end(), 0);
  auto lambda_print = [&](){
    for(auto ix: d){
      std::cout << ix << " ";
    }
    std::cout << std::endl;
  };
  
  lambda_print();
  ReactionA a(0.1);
  a.react(queue, d);
  lambda_print();
  ReactionB b(2);
  b.react(queue, d);
  lambda_print();
  
  // reset the data
  std::iota(d.begin(), d.end(), 0);

  // Now cast to the base reaction type and call through the react virtual function
  std::vector<std::shared_ptr<Reaction>> reactions(2);
  reactions[0] = make_reaction<ReactionA>(0.1);
  reactions[1] = make_reaction<ReactionB>(2);

  for(auto rx : reactions){
    rx->react(queue, d);
    lambda_print();
  }

  return 0;
}
