# def player_run(self):
#     game_over = False
#     self.generate_food()
#     self.fill_display()
#
#     xc, yc, xt, yt = 0, 0, 0, 0
#     while not game_over:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 game_over = True
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_LEFT:
#                     xt, yt = -10, 0
#                 elif event.key == pygame.K_RIGHT:
#                     xt, yt = 10, 0
#                 elif event.key == pygame.K_UP:
#                     xt, yt = 0, -10
#                 elif event.key == pygame.K_DOWN:
#                     xt, yt = 0, 10
#
#                 if event.key != self.opposite_dir[self.snake.curr_dir]:
#                     xc, yc = xt, yt
#                     self.snake.curr_dir = event.key
#
#         if self.snake.has_eaten(self.curr_food, xc, yc):
#             fx, fy = self.curr_food
#             self.snake.add_cell(fx, fy)
#             self.generate_food()
#         elif self.hasFailed(xc, yc):
#             game_over = True
#         else:
#             self.snake.update_body(xc, yc)
#
#         self.fill_display()
#         self.clock.tick(15)